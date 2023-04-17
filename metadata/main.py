#!/usr/bin/python3

"""
This script will take care of generating synonyms for the
dataField(s) that the user specifies.
"""

from typing import List
from urllib.parse import urljoin

from rich.prompt import Prompt
from rich import print
from requests import get

# Following variable should be updated according
# to the users requirements
INDEX_URL = "http://localhost:9200/test"  # Index to work on
FIELDS_TO_ENRICH = ["name"]  # Fields to enrich
# Suffix to add to the field names to store the enriched data
#
# If the field name is `title`, the field metadata will be stored at
# `title_metadata` if the suffix is set to `_metadata`
ENRICHED_FIELDS_SUFFIX = "_metadata"
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
        "enriched_suffix": {
            "name": "Suffix for enriched fields",
            "description": "Enter the suffix for enriched fields. If suffix is `_metadata` and field is `test` then final field will be `test_metadata`",
            "default": ENRICHED_FIELDS_SUFFIX,
            "validate_func": true_validate
        },
        "vector_time_df_name": {
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


def main():
    """
    Entry point into the script.

    This script requires python3.7+ due to a time requirement.
    """
    try:
        inputs = ask_user_inputs()
        print(inputs)
    except KeyboardInterrupt:
        # Exit gracefully!
        print("\nInterrupt received, exitting!")
        exit(0)
    except Exception as exc:
        print("\n[bold red]Exception occurred: ", exc)
        exit(-1)


if __name__ == "__main__":
    main()
