#!/usr/bin/python3

"""
This script will take care of generating synonyms for the
dataField(s) that the user specifies.
"""

from typing import List, Dict, Tuple
from urllib.parse import urljoin
from re import match
from json import loads
from time import time_ns, sleep

from rich.prompt import Prompt
from rich import print
from requests import get, put, post, delete

# Following variable should be updated according
# to the users requirements
INDEX_URL = "http://localhost:9200/test"  # Index to work on
FIELDS_TO_ENRICH = ["name"]  # Fields to enrich
# Field where the metadata of all the enriched fields will be stored
ENRICHED_FIELD_NAME = "metadata"
ENRICH_TIME_FIELD = "metadata_date"  # Field to indicate when enrichment happened


def get_index_alias(base_url: str, index_name: str) -> str:
    """
      Get the alias for the index. If the alias doesn't exist, we will
      return the passed index name.
      """
    alias_url = urljoin(base_url, f"_alias/{index_name}")
    alias_response = get(alias_url)

    if not alias_response.ok:
        return index_name

    # Return the first key inside the object returned
    return list(alias_response.json().keys())[0]


def get_mapping(source_url: str, source_index: str) -> Dict:
    """
      Return the mapping response for the passed index.
      """
    mappings_url = urljoin(source_url, source_index + "/_mappings")
    response = get(mappings_url)

    if not response.ok:
        raise Exception(
            "non OK response received while getting mappings of source index: ",
            response.status_code)

    mappings_json = response.json()
    return mappings_json


def validate_keys_in_mapping(source_url: str, source_index: str,
                             fields: List[str]) -> List[str]:
    """
      Validate that the passed keys are present in the mappings.

      This function returns a list of keys not present in the mappings
      """
    # The index exists, we can fetch the mappings and verify that the fields
    # are present
    mappings_json = get_mapping(source_url, source_index)

    mappings_object: Dict = mappings_json.get(source_index,
                                              {}).get("mappings",
                                                      {}).get("properties", {})
    property_keys: List = list(mappings_object.keys())

    # If even one field is not present, we will consider that a fatal error
    fields_not_present: List = []

    for field_to_vectorize in fields:
        if field_to_vectorize not in property_keys:
            fields_not_present.append(field_to_vectorize)

    return fields_not_present


def ping_url(url: str) -> bool:
    """
      Ping the passed URL and return OK if the request succeeds else return
      not ok.
      """
    try:
        response = get(url)
        return response.ok
    except Exception:
        return False


def ask_user_inputs():
    """
      Ask for the user inputs and make sure the inputs are validated before accepting
      them.

      We will take the following values as inputs
      - CLUSTER_URL
      - INDEX_NAME
      - FIELDS_TO_ENRICH
      - ENRICHED_FIELD_SUFFIX (will have default)
      - ENRICHED_TIME_FIELD (will have default)
      - OPENAI_KEY
      """
    values_to_return = {}

    def validate_url(input: str) -> bool:
        """
            Validate the URL
            """
        if input is None:
            return False

        return ping_url(input)

    def validate_index(input: str) -> bool:
        """
            Validate the index name.
            """
        if input is None:
            return False

        index_url = urljoin(values_to_return.get("cluster_url", ""), input)
        return ping_url(index_url)

    def validate_fields(input: str) -> bool:
        """
            Validate the input fields by making sure that they are present
            in the mappings.
            """
        if input is None:
            return False

        fields_array = input.split(",")

        cluster_url = values_to_return.get("cluster_url", "")
        index_passed = get_index_alias(cluster_url,
                                       values_to_return.get("index_name", ""))

        try:
            fields_not_present = validate_keys_in_mapping(cluster_url, index_passed,
                                                          fields_array)
            if not fields_not_present:
                return True

            print(
                f"[bold yellow]{', '.join(fields_not_present)}: not present in mappings"
            )
            return False
        except Exception as e:
            print("[bold yellow]Exception occurred: ", e)
            return False

    def true_validate(input: str) -> bool:
        return True if input is not None else False

    def parse_fields(input: str) -> List[str]:
        if input is None:
            return []

        return input.split(",")

    values_to_read = {
        "cluster_url": {
            "name": "Cluster URL",
            "description":
            "Enter the Cluster URL. Pass basic auth in the URL itself. Eg: http://localhost:9200",
            "validate_func": validate_url
        },
        "index_name": {
            "name": "Index Name",
            "description": "Enter the name of the index",
            "validate_func": validate_index
        },
        "fields_to_enrich": {
            "name": "Fields to Vectorize",
            "description":
            "Enter the fields to be enriched. Pass them separated by a comma (,). Eg: field1,field2,field3",
            "validate_func": validate_fields,
            "parser": parse_fields
        },
        "enriched_field_name": {
            "name": "Field name where the metadata will be stored",
            "description": "Enter the name of the field where the final metadata will be stored",
            "default": ENRICHED_FIELD_NAME,
            "validate_func": true_validate
        },
        "enrich_time_field": {
            "name": "Enriched time datafield name",
            "description":
            "Enter the field where the timestamp of when the field was enriched will be stored",
            "default": ENRICH_TIME_FIELD,
            "validate_func": true_validate
        },
        "open_ai_api_key": {
            "name": "OpenAI API Key",
            "description": "Enter the OpenAI API Key to access OpenAI endpoints",
            "validate_func": true_validate
        }
    }

    for key, value in values_to_read.items():
        # Build the input to show
        max_tries = 3
        try_each = 1
        while try_each <= max_tries:
            try_each += 1

            input_value = Prompt.ask(f"[blue]{value.get('description')}",
                                     default=value.get("default", None))

            if not value.get("validate_func")(input_value):
                print("[yellow]Input validation failed, please enter again!")
                continue

            # Since validation was fine, add this value for returning
            # and break this loop
            parser_func = value.get("parser", None)
            if parser_func is not None:
                values_to_return[key] = parser_func(input_value)
            else:
                values_to_return[key] = input_value

            break
        else:
            # Throw error indicating max tries reached.
            print("[bold red]Maximum tries reached for taking input, exiting!")
            exit(-1)

    return values_to_return


def get_settings(
    source_url: str,
    source_index: str,
    remove_keys: bool = False
) -> Dict:
    """
    Return the settings and remove pre-existing keys if the flag is passed
    """
    settings_url = urljoin(source_url, source_index + "/_settings")
    response = get(settings_url)

    if not response.ok:
        raise Exception(
            "non OK response received while getting settings of source index: ",
            response.status_code)

    settings_json = response.json().get(source_index,
                                        {}).get("settings", {}).get("index", {})

    if remove_keys:
        settings_json.pop("history", None)
        settings_json.pop("history.uuid", None)
        settings_json.pop("provided_name", None)
        settings_json.pop("uuid", None)
        settings_json.pop("version", None)
        settings_json.pop("creation_date", None)

    settings_json = {"index": settings_json}

    return settings_json


def reindexed_name(index_name: str) -> str:
    """
      Generate the re-indexed name of the index that needs to be
      setup.
      """
    is_match = match(".*reindexed_[0-9]+", index_name)

    if not is_match:
        return index_name + "_reindexed_1"

    # Since the index has _reindexed in it, we will need to extract
    # the digit and increment it.
    tokens = index_name.split("_")
    tokens[-1] = str(int(tokens[-1]) + 1)

    return "_".join(tokens)


def enriched_fields_setup(
    index_url: str,
    index_name: str,
    meta_field_name: str,
    meta_time_field_name: str
) -> Tuple[str, bool]:
    """
    Check if the source index has the metadata field
    and the other fields required. If it already has it,
    there is no need to do anything, we can just return
    the same index's name.

    If the index doesn't have it, we will need to create a
    temporary index that will follow the reindexed name
    structure used by RS. This index should have both the
    metadata fields added along with the settings of the
    original index.

    This function will return the name of the final index and
    a bool indicating whether a new index was created or not.
    """
    source_mappings = get_mapping(index_url, index_name)

    mappings = source_mappings.get(index_name, {}).get(
        "mappings", {})
    props = mappings.get("properties", {})
    prop_keys = props.keys()

    if meta_field_name in prop_keys and meta_time_field_name in prop_keys:
        # No need for a re-index, both fields are already present.
        # We can directly return the name of the index.
        return index_name, False

    # We will need to create the temporary index and add the
    temp_index = reindexed_name(index_name)

    # Pull the original settings of the temporary index
    original_settings = get_settings(index_url, index_name, True)

    # Inject the new mappings
    props[meta_field_name] = {
        "type": "text",
        "fields": {
            "autosuggest": {
                "type": "text",
                "analyzer": "autosuggest_analyzer",
                "search_analyzer": "standard"
            },
            "keyword": {
                "type": "keyword"
            },
            "search": {
                "type": "text",
                "analyzer": "ngram_analyzer",
                "search_analyzer": "standard"
            }
        }
    }

    props[meta_time_field_name] = {"type": "date"}
    mappings["properties"] = props

    create_body = {
        "mappings": mappings,
        "settings": original_settings
    }

    create_index_url = urljoin(index_url, temp_index)

    create_response = put(create_index_url, json=create_body)

    if not create_response.ok:
        err_msg = "error while creating index, got non OK response code: " + \
            str(create_response.status_code)
        print(err_msg)
        print("response received from create index: ", create_response.json())
        raise Exception(err_msg)

    return temp_index, True


def get_count_for_index(index_url: str) -> int:
    """
      Fetch the count for the passed index
      """
    count_URL = index_url + "/_count"
    count_response = get(count_URL)

    if not count_response.ok:
        raise Exception(f"could not determine total document count in \
                index, received: {count_response.status_code}")

    return count_response.json().get("count", 0)


def fetch_all_docs(index_url: str) -> List[Dict]:
    """
    Fetch all the documents from the source index in order to iterate through
    them and re-index them with the embeddings added.
    """

    def make_deep_page_call(search_url: str,
                            sort: List,
                            search_after: List = None) -> List:
        """
            Make the deep-pagination call based on the sort value
            """
        # If the search_after value is not present, we need to ignore it.
        search_body = {"size": 10000, "sort": sort}

        if search_after is not None:
            search_body["search_after"] = search_after

        search_response = post(search_url, json=search_body)

        if not search_response.ok:
            raise Exception(
                f"Search failed with response: {search_response.status_code} and \
                    response body: {search_response.json()}")

        response_json = search_response.json()
        return response_json.get("hits", {}).get("hits", [])

    # We will need to fetch the count and determine how many total docs
    # are present so that we can do a deep-pagination until we get
    # all the results
    total_count = get_count_for_index(index_url)
    total_hits = []

    # Determine the field we will use for sorting
    field_for_sorting = "_id"
    sort_body = [
        {
            field_for_sorting: "asc"
        },
    ]
    search_after = None
    search_url = index_url + "/_search"

    while True:
        # If we have fetched all the docs, stop!
        if len(total_hits) >= total_count:
            break

        # Make the deep pagination call
        hits_fetched = []
        try:
            hits_fetched = make_deep_page_call(
                search_url, sort_body, search_after)
        except Exception as e:
            print(f"Exception while fetching hits: {e}, trying again!")
            hits_fetched = make_deep_page_call(
                search_url, sort_body, search_after)

        total_hits.extend(hits_fetched)

        # Fetch the search_after value
        search_after = hits_fetched[-1].get("sort", None)

    # Once all the hits are fetched, return the hits
    return total_hits


def fetch_synonyms_from_open_ai(field_value: str, openai_key: str) -> List[str]:
    """
    Fetch the synonyms from OpenAI for the passed value.
    """
    if field_value == "":
        return []

    openai_body = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "system",
                "content": "You're a helpful assistant"
            },
            {
                "role": "user",
                "content": f"Give me 10 synonyms for '{field_value}' in JSON format."
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json"
    }

    openai_url = "https://api.openai.com/v1/chat/completions"
    openai_response = post(openai_url, headers=headers, json=openai_body)

    if not openai_response.ok:
        print("Got non OK response code from OpenAI: ",
              openai_response.status_code)
        return []

    response_json = openai_response.json()
    choices = response_json.get("choices", [])

    if not len(choices):
        return []

    response_as_str = choices[0].get("messages", {}).get("content", {})
    as_json = loads(response_as_str)
    return as_json.get("synonyms", [])


def index_document(
    base_url: str,
    index_name: str,
    doc: Dict,
    doc_id: str
) -> bool:
    """
    Index the document and consider upserting as well.

    We will also use the ID present in the older document
    """
    update_URL = urljoin(base_url, f"{index_name}/_update/{doc_id}")

    update_body = {"doc": doc, "doc_as_upsert": True}

    update_response = post(update_URL,
                           json=update_body,
                           headers={"Content-Type": "application/json"})

    if not update_response.ok:
        print("Failed to index document, got status code: ",
              update_response.status_code)
        print("Response received on failure is: ", update_response.json())
        return False

    return True


def main():
    """
    Entry point into the script.

    This script requires python3.7+ due to a time requirement.
    """
    try:
        inputs = ask_user_inputs()

        FIELDS_TO_ENRICH = inputs.get("fields_to_enrich", [])
        ENRICHED_FIELD_NAME = inputs.get("enriched_field_name", "")
        ENRICH_TIME_FIELD = inputs.get("enrich_time_field", "")

        source_index_url = inputs.get("cluster_url", "")
        og_index = inputs.get("index_name", "")
        source_index_name = get_index_alias(source_index_url, og_index)

        open_ai_api_key = inputs.get("open_ai_api_key", "")

        # There are two scenarios for the index field:
        #
        # - it already has the metadata field mapping
        # - it doesn't have the metadata field mapping.
        #
        # In the first case, we can just go ahead with the indexing
        # In the second case, we will need to re_index all the data
        # , delete the older index and then set an alias for the new
        # index.
        index_to_work_on, is_reindex = enriched_fields_setup(
            source_index_url,
            source_index_name,
            ENRICHED_FIELD_NAME,
            ENRICH_TIME_FIELD
        )

        # Fetch all the source docs and start working on them
        source_docs = fetch_all_docs(
            urljoin(source_index_url, source_index_name))

        for doc in source_docs:
            # Use all the input fields to get the synonyms, do them
            # one by one using the API.
            #
            # Create the doc in the index by doing an upsert.

            all_synonyms = []

            doc_source = doc.get("_source", {})
            doc_id = doc.get("_id", "")

            for field in FIELDS_TO_ENRICH:
                field_synonyms = fetch_synonyms_from_open_ai(
                    doc_source.get(field, ""),
                    open_ai_api_key
                )
                all_synonyms.extend(field_synonyms)

            doc_source[ENRICHED_FIELD_NAME] = all_synonyms
            doc_source[ENRICH_TIME_FIELD] = time_ns() // 1_000_000

            # Create the doc by upserting it.
            is_created = index_document(
                source_index_url,
                index_to_work_on,
                doc_source,
                doc_id,
            )

            if not is_created:
                print("Failed to index document with ID: ", doc_id)
            else:
                print("Indexed document with ID: ", doc_id)

        # Check if re-indexing was done and if so then we will
        # need to validate the count, delete the older index
        # and create an alias for the new one.

        # If it was not done, we don't need to do anything and can
        # exit right away.

        if not is_reindex:
            print("Exiting!")
            exit(0)

        # Verify the count matches between the new and the old index

        # Sleep for 2 seconds before verifying the count
        sleep(2)
        source_count = get_count_for_index(
            urljoin(source_index_url, source_index_name))

        dest_count = get_count_for_index(
            urljoin(source_index_url, index_to_work_on))

        if source_count != dest_count:
            raise Exception(
                f"source index count and destination index count doesn't match, \
                    aliasing will be skipped. Temporary index is: {index_to_work_on}"
            )

        # Delete the older index and create alias for the newer index
        delete_response = delete(urljoin(source_index_url, source_index_name))

        if not delete_response.ok:
            raise Exception(
                f"error while deleting source index to create alias with \
                    status: {delete_response.status_code} and json: {delete_response.json()}"
            )

        # Since it's deleted, create the alias and exit
        alias_url = urljoin(source_index_url, "_aliases")
        alias_body = {
            "actions": [{
                "add": {
                    "index": index_to_work_on,
                    "alias": og_index
                }
            }]
        }

        alias_response = post(alias_url,
                              headers={"Content-Type": "application/json"},
                              json=alias_body)

        if not alias_response.ok:
            raise Exception(
                f"error while creating alias with response: {alias_response.status_code} and \
                    json: {alias_response.json()}")

        print("Alias created successfully, script is complete!")

    except KeyboardInterrupt:
        # Exit gracefully!
        print("\nInterrupt received, exitting!")
        exit(0)
    except Exception as exc:
        print("\n[bold red]Exception occurred: ", exc)
        exit(-1)


if __name__ == "__main__":
    main()
