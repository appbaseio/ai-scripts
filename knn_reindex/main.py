#!/usr/bin/python3
"""
This script will handle converting/creating a kNN index
based on the source index passed.

It should be able to do this whole process without needing
user input except when the script starts.

This script requires the following things:
- Python3.7+
- `requests` package
"""

from requests import get, put, post, delete
from urllib.parse import urlsplit, urljoin
from typing import Dict, List
from time import time_ns, sleep
from datetime import datetime
from re import match

from rich.prompt import Prompt
from rich import print

# Global Vars to update before running the script
#
# Following vars should be updated before the script is
# run since these will decide the output and execution
# of the script.
SRC_INDEX = "http://localhost:9200/test"  # source index URL
DEST_INDEX = "http://localhost:9200/test"  # destination index URL
FIELDS_TO_VECTORIZE = ["Name", "Summary"]  # fields to vectorize and store
VECTOR_DF_NAME = "vector_data"  # where to store the vector data
# field to indicate when vectorization of data occurred
VECTOR_TIME_DF_NAME = "vector_added_at"
OPEN_AI_API_KEY = "sk-test"  # OpenAI API key


def does_index_exist(base_url: str, index_name: str) -> bool:
    """
      Check if the index exists or not
      """
    indices_url = urljoin(base_url, "_cat/indices")
    indices_response = get(indices_url, params={"format": "json"})

    if not indices_response.ok:
        raise Exception(
            "non OK response received from _cat/indices for the destination index: ",
            indices_response.status_code)

    # since response is valid
    response_as_json = indices_response.json()

    is_present = bool(
        len([i for i in response_as_json if i["index"] == index_name]))

    # If is_present, return
    if is_present:
        return is_present

    # As a fallback, also check aliases
    alias_url = urljoin(base_url, f"_alias/{index_name}")
    alias_response = get(alias_url)

    if alias_response.ok:
        return True

    return False


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


def create_index(base_url: str, index_name: str, source_mappings: Dict,
                 source_settings: Dict) -> bool:
    """
      Create the index with the necessary settings and mappings.
      """
    # Inject the vector data fields mappings
    source_mappings[VECTOR_DF_NAME] = {
        "type": "knn_vector",
        "dimension": 1536,
        "method": {
            "name": "hnsw",
            "space_type": "cosinesimil",
            "engine": "nmslib"
        }
    }

    # Inject the vector datefield mappings
    source_mappings[VECTOR_TIME_DF_NAME] = {"type": "date"}

    # Inject the knn settings
    source_settings["knn"] = True
    source_settings["knn.algo_param.ef_search"] = 100

    create_index_body = {
        "settings": source_settings,
        "mappings": {
            "properties": source_mappings
        }
    }

    create_index_URL = urljoin(base_url, index_name)

    create_index_response = put(create_index_URL, json=create_index_body)

    if not create_index_response.ok:
        print("error while creating index, got non OK status code: ",
              create_index_response.status_code)
        print("response received for put request is: ",
              create_index_response.json())
        return False

    return True


def dest_setup(dest_url: str, dest_index: str, source_mappings: Dict,
               source_settings: Dict):
    """
      Take care of setup of various things in the destination
      index. This should include:
      - verifying that `knn` is set in settings
      - verifying that the vector data field is set with proper
      mapping.
      - verifying that the vector_added_at field is set with proper
      mapping.
      """
    # Check if the index already exists
    is_index_present = does_index_exist(dest_url, dest_index)

    if not is_index_present:
        # Create the index with the settings and mappings
        is_created = create_index(dest_url, dest_index, source_mappings,
                                  source_settings)
        if not is_created:
            raise Exception("error while creating index!")

        return

    # If the index is present, we will need to check settings
    # and mappings to verify that the vector data fields are
    # present

    settings_url = urljoin(dest_url, dest_index + "/_settings")
    response = get(settings_url)

    if not response.ok:
        raise Exception(
            "non OK response received while getting settings of destination index: ",
            response.status_code)

    # If ok response was received, we can verify if `knn` is enabled
    # or not.
    response_json = response.json()

    is_knn = response_json.get(dest_index, {}).get("settings",
                                                   {}).get("index",
                                                           {}).get("knn", False)
    if not is_knn:
        # We cannot do anything about it.
        raise Exception(
            "`knn` is not enabled on the index. This script cannot enable it automatically since \
                that will require a the index to be re-created.")

    # We will now need to check the mappings to confirm that the
    # vector data field and the field to set the vector creation time
    # is set.
    mappings_url = urljoin(dest_url, dest_index + "/_mappings")
    response = get(mappings_url)

    if not response.ok:
        raise Exception(
            "non OK response received while getting mappings of destination index: ",
            response.status_code)

    mappings_json = response.json()

    vector_df = mappings_json.get(dest_index,
                                  {}).get("mappings",
                                          {}).get("properties",
                                                  {}).get(VECTOR_DF_NAME, {})
    vector_df_dtype = vector_df.get("type", "")
    vector_df_dimension = vector_df.get("dimension", 0)

    if vector_df_dtype != "knn_vector":
        raise Exception(
            "vector field has non vector datatype, cannot continue!")

    # 1536 is the required dimension to store data received from OpenAI
    if vector_df_dimension != 1536:
        raise Exception(
            "vector dimension does not match required dimension of `1536`!")


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


def get_settings(source_url: str,
                 source_index: str,
                 remove_keys: bool = False) -> Dict:
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


def fetch_embeddings_from_open_ai(text_to_embed: str, api_key: str) -> List:
    """
      Fetch the embeddings of the passed data from OpenAI
      """
    embeddings_URL = "https://api.openai.com/v1/embeddings"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    body_to_send = {"model": "text-embedding-ada-002", "input": text_to_embed}

    embeddings_response = post(embeddings_URL,
                               headers=headers,
                               json=body_to_send)

    if not embeddings_response.ok:
        raise Exception(
            f"error while fetching embeddings with response: {embeddings_response.status_code} \
                and response json: {embeddings_response.json()}")

    return embeddings_response.json().get("data", [])[0].get("embedding", [])


def inject_embeddings(doc_source: Dict, api_key: str) -> Dict:
    """
      Fetch the embeddings based on the passed document
      """
    # Extract all the keys that are to be used for getting the embeddings
    embedding_inputs = []

    for field_to_vectorize in FIELDS_TO_VECTORIZE:
        embedding_inputs.append(doc_source.get(field_to_vectorize, ""))

    embedding_text = ", ".join(embedding_inputs)

    open_ai_embeddings = fetch_embeddings_from_open_ai(embedding_text, api_key)

    doc_source[VECTOR_DF_NAME] = open_ai_embeddings
    doc_source[VECTOR_TIME_DF_NAME] = time_ns() // 1_000_000

    return doc_source


def index_document(base_url: str, index_name: str, doc: Dict,
                   doc_id: str) -> bool:
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
      - FIELDS_TO_VECTORIZE
      - VECTOR_DF_NAME (will have default)
      - VECTOR_DF_TIME_NAME (will have default)
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
        "fields_to_vectorize": {
            "name": "Fields to Vectorize",
            "description":
            "Enter the fields to be vectorized. Pass them separated by a comma (,). Eg: field1,field2,field3",
            "validate_func": validate_fields,
            "parser": parse_fields
        },
        "vector_df_name": {
            "name": "Vector DataField Name",
            "description": "Enter the name of the vector datafield",
            "default": VECTOR_DF_NAME,
            "validate_func": true_validate
        },
        "vector_time_df_name": {
            "name": "Vector Time DataField Name",
            "description":
            "Enter the field where the timestamp of when the vector data is added will be stored",
            "default": VECTOR_TIME_DF_NAME,
            "validate_func": true_validate
        },
        "open_ai_api_key": {
            "name": "OpenAI API Key",
            "description": "Enter the OpenAI API Key to access OpenAI endpoints",
            "validate_func": true_validate,
            "is_password": True
        }
    }

    for key, value in values_to_read.items():
        # Build the input to show
        max_tries = 3
        try_each = 1
        while try_each <= max_tries:
            try_each += 1

            input_value = Prompt.ask(f"[blue]{value.get('description')}",
                                     default=value.get("default", None),
                                     password=value.get("is_password", False))

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


def main():
    """
      Entry point into the script.

      This script requires python3.7+ due to a time requirement
      """
    # Take the inputs
    try:
        inputs = ask_user_inputs()

        SRC_INDEX = urljoin(inputs.get("cluster_url", ""),
                            inputs.get("index_name", ""))
        DEST_INDEX = SRC_INDEX
        FIELDS_TO_VECTORIZE = inputs.get("fields_to_vectorize", [])
        VECTOR_DF_NAME = inputs.get("vector_df_name", "")
        VECTOR_TIME_DF_NAME = inputs.get("vector_time_df_name", "")
        OPEN_AI_API_KEY = inputs.get("open_ai_api_key", "")

        # Indicates if the name passed by the user is an alias
        # or not.
        is_passed_name_alias = False

        # Split the source and destination URL's
        source_URL_splitted = urlsplit(SRC_INDEX)
        source_URL_base = f"{source_URL_splitted.scheme}://{source_URL_splitted.netloc}"
        # Don't need the prefix `/`
        older_index = source_URL_splitted.path[1:]
        source_URL_index = get_index_alias(source_URL_base,
                                           source_URL_splitted.path[1:])

        if source_URL_index != older_index:
            is_passed_name_alias = True

        dest_URL_splitted = urlsplit(DEST_INDEX)
        dest_URL_base = f"{dest_URL_splitted.scheme}://{dest_URL_splitted.netloc}"
        og_index = dest_URL_splitted.path[1:]
        dest_URL_index = get_index_alias(dest_URL_base, og_index)

        # Check if the source and destination index is the same
        is_same_index = SRC_INDEX == DEST_INDEX
        is_reindexing = False

        # Create a temp index that will be used if we need to re-index
        # the same
        temp_index = reindexed_name(dest_URL_index)

        # NOTE: We don't need to validate the source index since that will be validated
        # when the user input is taken!
        mappings_json = get_mapping(source_URL_base, source_URL_index)
        source_mappings: Dict = mappings_json.get(source_URL_index, {}).get(
            "mappings", {}).get("properties", {})

        source_settings = get_settings(source_URL_base, source_URL_index, True)

        # Do the pre-setup of the destination index
        try:
            dest_setup(dest_URL_base, dest_URL_index, source_mappings,
                       source_settings)
        except Exception as dest_exception:
            # We don't need to throw an error if the source index and the
            # destination index is the same.
            if not is_same_index:
                raise dest_exception

            # Create the temp index that will be connected to the
            # actual index as alias
            is_reindexing = True
            dest_URL_index = temp_index
            dest_setup(dest_URL_base, dest_URL_index, source_mappings,
                       source_settings)

        # Fetch all the source docs
        source_docs = fetch_all_docs(SRC_INDEX)

        for doc in source_docs:
            source_obj = doc.get("_source", {})
            doc_id = doc.get("_id", "")

            # If the doc contains the vector_added_at field then make sure that
            # the field is more than 2 months old or skip the doc.
            if VECTOR_TIME_DF_NAME in source_obj:
                time_in_ms = source_obj.get(VECTOR_TIME_DF_NAME)
                added_at = datetime.fromtimestamp(time_in_ms / 1000)
                current_d = datetime.now()

                if (current_d - added_at).seconds < 2 * 30 * 86400:
                    print("Skipping doc since vector was added less than 2 months ago: ",
                          doc_id)
                    continue

            doc_with_injection = inject_embeddings(source_obj, OPEN_AI_API_KEY)

            # Make the PUT call that will consider upserting as well
            is_created = index_document(dest_URL_base, dest_URL_index,
                                        doc_with_injection, doc_id)
            if not is_created:
                print("Failed to index document with ID: ", doc_id)
            else:
                print("Indexed document with ID: ", doc_id)

        if not is_reindexing:
            print("Exiting!")
            exit(0)

        # If it is the same index, we will need to delete the older
        # one and set alias for the new one

        # Verify the count matches between the new and the old index

        # Sleep for 2 seconds before verifying the count
        sleep(2)
        source_count = get_count_for_index(SRC_INDEX)

        new_dest_url = urljoin(dest_URL_base, dest_URL_index)
        dest_count = get_count_for_index(new_dest_url)

        if source_count != dest_count:
            raise Exception(
                f"source index count and destination index count doesn't match, \
                    aliasing will be skipped. Temporary index is: {dest_URL_index}"
            )

        # Delete the older index and create alias for the newer index
        delete_response = delete(source_URL_base + "/" + source_URL_index)

        if not delete_response.ok:
            raise Exception(
                f"error while deleting source index to create alias with \
                    status: {delete_response.status_code} and json: {delete_response.json()}"
            )

        # Since it's deleted, create the alias and exit
        alias_url = urljoin(dest_URL_base, "_aliases")
        alias_body = {
            "actions": [{
                "add": {
                    "index": temp_index,
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
