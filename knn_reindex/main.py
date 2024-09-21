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
from packaging import version
from urllib.parse import urlsplit, urljoin
from typing import Dict, List
from time import time_ns, sleep
from datetime import datetime
from re import match
import json
import os

from rich.prompt import Prompt
from rich import print

# Global Vars to update before running the script
#
# Following vars should be updated before the script is
# run since these will decide the output and execution
# of the script.
SRC_INDEX = "http://localhost:9200/test"  # source index URL
FIELDS_TO_VECTORIZE = ["name", "one_liner", "long_description"]  # fields to vectorize and store
VECTOR_DF_NAME = "vector_data"  # where to store the vector data
# field to indicate when vectorization of data occurred
VECTOR_TIME_DF_NAME = "vector_added_at"
OPEN_AI_API_KEY = "sk-test"  # OpenAI API key
DEMO_MODE = True  # Default to demo mode

CONFIG_FILE = "config.json"

def save_user_inputs(inputs: dict):
    """
    Save user inputs to a config file (JSON format)
    """
    with open(CONFIG_FILE, "w") as config_file:
        json.dump(inputs, config_file, indent=4)

def load_user_inputs() -> dict:
    """
    Load user inputs from the config file if it exists
    """
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as config_file:
            return json.load(config_file)
    return {}


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


def get_search_engine_version(base_url: str) -> str:
    """
    Get the version of the search engine (Elasticsearch or OpenSearch)
    """
    response = get(base_url)
    if not response.ok:
        raise Exception("Failed to get version information")
    version_info = response.json().get("version", {}).get("number", "")
    return version_info


def update_index_mapping(base_url: str, index_name: str, is_opensearch: bool) -> bool:
    """
    Update the index mapping to include vector data fields and timestamp field.
    Handle cases where the vector type is already correctly set.
    """
    properties = {
        VECTOR_DF_NAME: {
            "type": "knn_vector" if is_opensearch else "dense_vector"
        },
        VECTOR_TIME_DF_NAME: {
            "type": "date",
            "format": "epoch_millis"
        }
    }

    # Add dimension or dims based on search engine
    if is_opensearch:
        properties[VECTOR_DF_NAME]["dimension"] = 1536
        properties[VECTOR_DF_NAME]["method"] = {
            "name": "hnsw",
            "engine": "lucene"
        }
    else:
        properties[VECTOR_DF_NAME]["dims"] = 1536

    mapping_update_body = {
        "properties": properties
    }

    mapping_url = urljoin(base_url, f"{index_name}/_mapping")
    mapping_response = put(mapping_url, json=mapping_update_body)

    if not mapping_response.ok:
        # Check for specific error regarding type conflict
        response_json = mapping_response.json()
        error = response_json.get('error', {})
        root_cause = error.get('root_cause', [{}])[0]
        if (error.get('type') == 'illegal_argument_exception' and 
            ('cannot be changed from type [knn_vector] to [knn_vector]' in root_cause.get('reason', '') or
             f'Mapper for [{VECTOR_DF_NAME}] conflicts with existing mapper' in root_cause.get('reason', ''))):
            # Log that the type is already correctly set
            print(f"[yellow]Mapping already set correctly for {VECTOR_DF_NAME}, no changes needed.")
            return True
        else:
            print("Error while updating index mapping, got non-OK status code: ",
                  mapping_response.status_code)
            print("Response received for put request is: ", response_json)
            return False

    return True


def create_index(base_url: str, index_name: str, source_mappings: Dict,
                 source_settings: Dict, is_opensearch: bool) -> bool:
    """
      Create the index with the necessary settings and mappings.
      """
    # Inject the vector data fields mappings
    if is_opensearch:
        source_mappings[VECTOR_DF_NAME] = {
            "type": "knn_vector",
            "dimension": 1536,
            "method": {
                "name": "hnsw",
                "engine": "lucene"
            }
        }
    else:
        source_mappings[VECTOR_DF_NAME] = {
            "type": "dense_vector",
            "dims": 1536
        }

    # Inject the vector datefield mappings
    source_mappings[VECTOR_TIME_DF_NAME] = {
        "type": "date", "format": "epoch_millis"}

    # Inject the knn settings
    if is_opensearch:
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


def fetch_all_docs(index_url: str, demo_mode: bool) -> List[Dict]:
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
        if demo_mode:
            search_body = {"size": 10}

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
    field_for_sorting = "id"
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

        if demo_mode and len(total_hits) >= 10:
            total_hits = total_hits[:10]
            break

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

    body_to_send = {"model": "text-embedding-3-small", "input": text_to_embed}

    embeddings_response = post(embeddings_URL,
                               headers=headers,
                               json=body_to_send)

    if not embeddings_response.ok:
        raise Exception(
            f"error while fetching embeddings with response: {embeddings_response.status_code} \
                and response json: {embeddings_response.json()}")

    return embeddings_response.json().get("data", [])[0].get("embedding", [])


def inject_embeddings(doc_source: Dict, fields_to_vectorize: list[str], api_key: str) -> Dict:
    """
      Fetch the embeddings based on the passed document
      """
    # Extract all the keys that are to be used for getting the embeddings
    embedding_inputs = []

    for field_to_vectorize in fields_to_vectorize:
        # Replace None with empty string
        value = doc_source.get(field_to_vectorize, "")
        if value is None:
            value = ""
        embedding_inputs.append(value)

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
    them. If previously entered inputs exist, load them from the config file and
    offer the option to reuse or modify.

    We will take the following values as inputs:
      - CLUSTER_URL (source and dest)
      - INDEX_NAME  (source and dest)
      - FIELDS_TO_VECTORIZE
      - VECTOR_DF_NAME (will have default)
      - VECTOR_DF_TIME_NAME (will have default)
      - OPENAI_KEY
      - DEMO_MODE (default to True)
      - DEST_INDEX (optional)
    """
    values_to_return = {}

    # Load previously saved inputs if available
    saved_inputs = load_user_inputs()
    if saved_inputs:
        print("[green]Previous configuration found. Would you like to use it?")
        use_saved = Prompt.ask("Use previously entered values?", choices=["yes", "no"], default="yes")
        if use_saved == "yes":
            return saved_inputs

    # If not using saved values, ask for inputs as before
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

        cluster_url = values_to_return.get("cluster_url", "")
        if not cluster_url.endswith("/"):
            cluster_url += "/"
        index_url = urljoin(cluster_url, input)
        print("pinging index_url: ", index_url)
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
            "Enter the Cluster URL. Pass basic auth creds in the URL itself. e.g. URL format: https://$user:$pass@mydomain.com, http://localhost:9200",
            "validate_func": validate_url
        },
        "index_name": {
            "name": "Index Name",
            "description": "Enter the name of the source index",
            "validate_func": validate_index
        },
        "destination_cluster_url": {
            "name": "Destination Cluster URL",
            "description":
            "Enter the Destination Cluster URL (defaults to source cluster URL)",
            "validate_func": validate_url
        },
        "dest_index_name": {
            "name": "Destination Index Name",
            "description": "Enter the name of the destination index (defaults to source index name)",
            "default": "",
            "validate_func": true_validate
        },
        "fields_to_vectorize": {
            "name": "Fields to Vectorize",
            "description":
            "Enter the fields to be vectorized. Pass them separated by a comma (,). Eg: field1,field2,field3",
            "default": ",".join(FIELDS_TO_VECTORIZE),
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
            "default": OPEN_AI_API_KEY,
            "validate_func": true_validate,
            "is_password": True
        },
        "demo_mode": {
            "name": "Demo Mode",
            "description": "Enable demo mode? This will only index 10 documents, enter false for production indexing mode",
            "default": "true",
            "validate_func": lambda x: x.lower() in ["true", "false"],
            "parser": lambda x: x.lower() == "true"
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

    # Save the inputs to config.json
    save_user_inputs(values_to_return)
    return values_to_return


def process_and_index_document(doc: Dict, fields_to_vectorize: List[str], vector_time_df_name: str, 
                               open_ai_api_key: str, dest_url_base: str, dest_index: str) -> bool:
    """
    Process a single document: inject embeddings and index it. 
    If vector data already exists and is recent, skip embedding injection.
    """
    source_obj = doc.get("_source", {})
    doc_id = doc.get("_id", "")

    # Check if the vector was added in the last 90 days
    if vector_time_df_name in source_obj:
        time_in_ms = source_obj.get(vector_time_df_name)
        added_at = datetime.fromtimestamp(time_in_ms / 1000)
        current_d = datetime.now()

        if (current_d - added_at).days < 90:
            print("Skipping fetching of OpenAI embeddings since vector was added less than 90 days ago: ", doc_id)
            # Directly index the document without injecting embeddings
            return index_document(dest_url_base, dest_index, source_obj, doc_id)

    # Inject embeddings
    doc_with_injection = inject_embeddings(source_obj, fields_to_vectorize, open_ai_api_key)

    # Index the document with injected embeddings
    return index_document(dest_url_base, dest_index, doc_with_injection, doc_id)


def main():
    """
    Entry point into the script.
    """
    try:
        inputs = ask_user_inputs()

        SRC_INDEX = urljoin(inputs.get("cluster_url", ""), inputs.get("index_name", ""))
        DEST_INDEX = urljoin(inputs.get("destination_cluster_url", inputs.get("cluster_url", "")),
                             inputs.get("dest_index_name", "") or inputs.get("index_name", ""))
        FIELDS_TO_VECTORIZE = inputs.get("fields_to_vectorize", [])
        VECTOR_DF_NAME = inputs.get("vector_df_name", "")
        VECTOR_TIME_DF_NAME = inputs.get("vector_time_df_name", "")
        OPEN_AI_API_KEY = inputs.get("open_ai_api_key", "")
        DEMO_MODE = inputs.get("demo_mode", True)

        print("fields_to_vectorize: ", FIELDS_TO_VECTORIZE)
        print("dest_index_name: ", DEST_INDEX)

        # split the source URL to get the base and the index
        source_URL_splitted = urlsplit(SRC_INDEX)
        source_URL_base = f"{source_URL_splitted.scheme}://{source_URL_splitted.netloc}"
        source_URL_index = get_index_alias(source_URL_base, source_URL_splitted.path[1:])

        # split the destination URL to get the base and the index
        dest_URL_splitted = urlsplit(DEST_INDEX)
        dest_URL_base = f"{dest_URL_splitted.scheme}://{dest_URL_splitted.netloc}"
        dest_URL_index = get_index_alias(dest_URL_base, dest_URL_splitted.path[1:])

        # Get the version of the search engine
        s_version = get_search_engine_version(dest_URL_base)
        is_opensearch = s_version.startswith("2.") or s_version.startswith("7.")

        if s_version.startswith("8.") and version.parse(s_version) < version.parse("8.12.0") and not is_opensearch:
            raise Exception("Elasticsearch versions below 8.12.0 do not support vector indexing with 1536 dimensions. Please upgrade to the latest version.")

        # If the destination index is different and does not exist, create it
        if DEST_INDEX != SRC_INDEX and not does_index_exist(dest_URL_base, dest_URL_index):
            source_mappings = get_mapping(source_URL_base, source_URL_index).get(source_URL_index, {}).get("mappings", {}).get("properties", {})
            source_settings = get_settings(source_URL_base, source_URL_index, True)
            if not create_index(dest_URL_base, dest_URL_index, source_mappings, source_settings, is_opensearch):
                raise Exception("Failed to create destination index")

        # Update the index mapping to include vector and timestamp fields
        if not update_index_mapping(dest_URL_base, dest_URL_index, is_opensearch):
            raise Exception("Failed to update index mapping")

        # Fetch all the source docs
        source_docs = fetch_all_docs(SRC_INDEX, DEMO_MODE)

        # Process and index each document
        index_count = 0
        for doc in source_docs:
            if process_and_index_document(doc, FIELDS_TO_VECTORIZE, VECTOR_TIME_DF_NAME, OPEN_AI_API_KEY, dest_URL_base, dest_URL_index):
                index_count += 1
                print(f"{index_count} Indexed document with ID: {doc.get('_id', '')}")
            else:
                print(f"Failed to index document with ID: {doc.get('_id', '')}")

        print("Indexing complete!")
    except KeyboardInterrupt:
        # Exit gracefully
        print("\nInterrupt received, exiting!")
        exit(0)
    except Exception as exc:
        print("\n[bold red]Exception occurred: ", exc)
        exit(-1)

if __name__ == "__main__":
    main()
