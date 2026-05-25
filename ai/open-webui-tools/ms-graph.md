## About

We want to call [Microsoft Graph API](https://learn.microsoft.com/en-us/graph/overview).

## Supported Features

The following Graph API endpoints are supported:
 - [List teams](https://learn.microsoft.com/en-us/graph/api/teams-list?view=graph-rest-1.0&tabs=http)

## Assumption

We are just a Microsoft user and don't know the Azure application's client secret.

## Setup

Since we don't know our organization's Azure applications information, we will use [OAuth 2.0 Device Code Flow](https://learn.microsoft.com/en-us/entra/identity-platform/v2-oauth2-device-code).

To obtain the access token, we can start with a command like this:
```sh
curl -X POST "https://login.microsoftonline.com/organizations/oauth2/v2.0/devicecode" -H "Content-Type: application/x-www-form-urlencoded" -d "client_id=1fec8e78-bce4-4aaf-ab1b-5451cc387264&scope=https://graph.microsoft.com/Team.ReadBasic.All offline_access" 
```

Note that the client ID `1fec8e78-bce4-4aaf-ab1b-5451cc387264` is a globally available Azure application configured by Microsoft. That application is used by Microsoft Power Apps and PowerShell.

By the way, we can also use the client ID `de8bc8b5-d9f9-48b1-a8ad-b748da725064`, which is another globally available application used by Microsoft Office and Graph Explorer.

The response body would look like this:
```json
{
  "user_code": "OXJ...",
  "device_code": "OBg...",
  "verification_uri": "https://login.microsoft.com/device",
  "expires_in": 900,
  "interval": 5,
  "message": "To sign in, use a web browser to open the page https://login.microsoft.com/device and enter the code OXJ... to authenticate." 
}
```

Follow the instructions in the attribute `message` of the payload.

After that, run a command like this:
```sh
curl -X POST "https://login.microsoftonline.com/organizations/oauth2/v2.0/token" -H "Content-Type: application/x-www-form-urlencoded" -d "grant_type=urn:ietf:params:oauth:grant-type:device_code&client_id=1fec8e78-bce4-4aaf-ab1b-5451cc387264&device_code=OBg..." 
```

The response body would look like this:
```json
{
  "token_type": "Bearer",
  "scope": "email openid profile https://graph.microsoft.com/AppCatalog.Read.All https://graph.microsoft.com/Calendars.Read https://graph.microsoft.com/Calendars.Read.Shared https://graph.microsoft.com/Calendars.ReadWrite https://graph.microsoft.com/Calendars.ReadWrite.Shared https://graph.microsoft.com/Channel.ReadBasic.All https://graph.microsoft.com/ChatMessage.Send https://graph.microsoft.com/Contacts.ReadWrite.Shared https://graph.microsoft.com/Files.ReadWrite.All https://graph.microsoft.com/FileStorageContainer.Selected https://graph.microsoft.com/Group.Read.All https://graph.microsoft.com/InformationProtectionPolicy.Read https://graph.microsoft.com/Mail.ReadWrite https://graph.microsoft.com/Mail.Send https://graph.microsoft.com/MailboxSettings.ReadWrite https://graph.microsoft.com/Notes.ReadWrite.All https://graph.microsoft.com/Organization.Read.All https://graph.microsoft.com/People.Read https://graph.microsoft.com/Place.Read.All https://graph.microsoft.com/Sites.ReadWrite.All https://graph.microsoft.com/Tasks.ReadWrite https://graph.microsoft.com/Team.ReadBasic.All https://graph.microsoft.com/TeamsActivity.Send https://graph.microsoft.com/TeamsAppInstallation.ReadForTeam https://graph.microsoft.com/TeamsTab.Create https://graph.microsoft.com/User.ReadBasic.All",
  "expires_in": 5099,
  "ext_expires_in": 5099,
  "access_token": "eyJ...",
  "refresh_token": "1.A...",
  "foci": "1" 
}
```

By the way, we can refresh the token like this:
```sh
curl -X POST "https://login.microsoftonline.com/organizations/oauth2/v2.0/token" -H "Content-Type: application/x-www-form-urlencoded" -d "grant_type=refresh_token&client_id=1fec8e78-bce4-4aaf-ab1b-5451cc387264&refresh_token=1.A...&scope=https://graph.microsoft.com/Team.ReadBasic.All offline_access" 
```

The response body would look like this:
```json
{
  "token_type": "Bearer",
  "scope": "email openid profile https://graph.microsoft.com/AppCatalog.Read.All https://graph.microsoft.com/Calendars.Read https://graph.microsoft.com/Calendars.Read.Shared https://graph.microsoft.com/Calendars.ReadWrite https://graph.microsoft.com/Calendars.ReadWrite.Shared https://graph.microsoft.com/Channel.ReadBasic.All https://graph.microsoft.com/ChatMessage.Send https://graph.microsoft.com/Contacts.ReadWrite.Shared https://graph.microsoft.com/Files.ReadWrite.All https://graph.microsoft.com/FileStorageContainer.Selected https://graph.microsoft.com/Group.Read.All https://graph.microsoft.com/InformationProtectionPolicy.Read https://graph.microsoft.com/Mail.ReadWrite https://graph.microsoft.com/Mail.Send https://graph.microsoft.com/MailboxSettings.ReadWrite https://graph.microsoft.com/Notes.ReadWrite.All https://graph.microsoft.com/Organization.Read.All https://graph.microsoft.com/People.Read https://graph.microsoft.com/Place.Read.All https://graph.microsoft.com/Sites.ReadWrite.All https://graph.microsoft.com/Tasks.ReadWrite https://graph.microsoft.com/Team.ReadBasic.All https://graph.microsoft.com/TeamsActivity.Send https://graph.microsoft.com/TeamsAppInstallation.ReadForTeam https://graph.microsoft.com/TeamsTab.Create https://graph.microsoft.com/User.ReadBasic.All",
  "expires_in": 5050,
  "ext_expires_in": 5050,
  "access_token": "eyJ...",
  "refresh_token": "1.AV...",
  "foci": "1" 
}
```

After the authentication, save the access token payload as a file.

For example, we can do the following:
```sh
mkdir -p /app/backend/data/
vim /app/backend/data/graph-oauth.json
```

The content should look like this:
```json
{
  "token_type": "Bearer",
  "scope": "email openid profile https://graph.microsoft.com/AppCatalog.Read.All https://graph.microsoft.com/Calendars.Read https://graph.microsoft.com/Calendars.Read.Shared https://graph.microsoft.com/Calendars.ReadWrite https://graph.microsoft.com/Calendars.ReadWrite.Shared https://graph.microsoft.com/Channel.ReadBasic.All https://graph.microsoft.com/ChatMessage.Send https://graph.microsoft.com/Contacts.ReadWrite.Shared https://graph.microsoft.com/Files.ReadWrite.All https://graph.microsoft.com/FileStorageContainer.Selected https://graph.microsoft.com/Group.Read.All https://graph.microsoft.com/InformationProtectionPolicy.Read https://graph.microsoft.com/Mail.ReadWrite https://graph.microsoft.com/Mail.Send https://graph.microsoft.com/MailboxSettings.ReadWrite https://graph.microsoft.com/Notes.ReadWrite.All https://graph.microsoft.com/Organization.Read.All https://graph.microsoft.com/People.Read https://graph.microsoft.com/Place.Read.All https://graph.microsoft.com/Sites.ReadWrite.All https://graph.microsoft.com/Tasks.ReadWrite https://graph.microsoft.com/Team.ReadBasic.All https://graph.microsoft.com/TeamsActivity.Send https://graph.microsoft.com/TeamsAppInstallation.ReadForTeam https://graph.microsoft.com/TeamsTab.Create https://graph.microsoft.com/User.ReadBasic.All",
  "expires_in": 5099,
  "ext_expires_in": 5099,
  "access_token": "eyJ...",
  "refresh_token": "1.A...",
  "foci": "1" 
}
```

We recommend adjusting the permission like this:
```sh
chmod o+rw /app/backend/data/graph-oauth.json
```

We are set!

Fortunately, this operation is one-time only.

Once we create the access token file, the Python code in the section below will use the refresh token and update the file automatically.

When we apply the Python code in the section below, we should configure valves, especially `OAUTH_JSON_PATH`.

## Code

```py
import json

from pydantic import BaseModel, Field
import requests
from typing import Dict, Any


class Tools:
    class Valves(BaseModel):
        OAUTH_JSON_PATH: str = Field(
            default="/app/backend/data/graph-oauth.json",
            description="Path to OAuth payload, which contains the access token and refresh token.",
        )
        GRAPH_CLIENT_ID: str = Field(
            default="1fec8e78-bce4-4aaf-ab1b-5451cc387264",
            description="Azure client ID for Graph API.",
        )
        MAX_RESPONSE_SIZE: int = Field(
            default=67000, description="The response from the Graph API can be too large for the model. This valve sets the maximum response size (in characters) that will be returned to the model. If the API response exceeds this size, it will return an error message suggesting limiting the size of the page."
        )

    def __init__(self):
        self.valves = self.Valves()

    def _load_oauth_payload(self) -> Dict[str, Any]:
        """
        Load and parse the OAuth payload from the JSON file.

        :return: Dictionary containing OAuth data (access_token, refresh_token, scope, etc.)
        :raises FileNotFoundError: If the OAuth JSON file does not exist.
        :raises json.JSONDecodeError: If the file is not valid JSON.
        """
        with open(self.valves.OAUTH_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_oauth_payload(self, oauth_data: Dict[str, Any]) -> None:
        """
        Save the OAuth payload to the JSON file, replacing the existing content.

        :param oauth_data: Dictionary containing OAuth data to persist.
        """
        with open(self.valves.OAUTH_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(oauth_data, f, indent=2)

    def _common_http_headers(self) -> Dict[str, str]:
        """
        Build HTTP headers for Graph API requests, including the Authorization header with the current access token.

        :return: Dictionary of HTTP headers (Authorization and Accept).
        :raises Exception: If unable to load or parse the OAuth payload or retrieve the access token.
        """
        oauth_data = self._load_oauth_payload()
        token_type = oauth_data.get("token_type", "Bearer")
        access_token = oauth_data.get("access_token", "")
        if not access_token:
            raise ValueError("No access_token found in OAuth payload")
        return {
            "Authorization": f"{token_type} {access_token}",
            "Accept": "application/json",
        }

    def _is_token_expired_response(self, response: requests.Response) -> bool:
        """
        Check if the Graph API response indicates an authentication/token failure.

        :param response: The response object from a Graph API request.
        :return: True if the response indicates token expiry (401 status), False otherwise.
        """
        return response.status_code == 401

    def _refresh_graph_access_token(self) -> None:
        """
        Refresh the Graph API access token using the refresh token from the OAuth payload.

        This method:
        1. Loads the OAuth payload from disk.
        2. Extracts the refresh_token and scope.
        3. POSTs to the Microsoft token endpoint with refresh_token grant.
        4. Merges the new token response with the existing OAuth payload.
        5. Saves the updated payload back to disk.

        :raises Exception: If unable to load OAuth, missing refresh_token, or token refresh fails.
        """
        oauth_data = self._load_oauth_payload()

        refresh_token = oauth_data.get("refresh_token")
        scope = oauth_data.get("scope")

        if not refresh_token:
            raise ValueError("Missing refresh_token in OAuth payload")

        form_data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.valves.GRAPH_CLIENT_ID,
        }
        if scope:
            form_data["scope"] = scope

        try:
            refresh_response = requests.post(
                "https://login.microsoftonline.com/organizations/oauth2/v2.0/token",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data=form_data,
                timeout=10,
            )
            if refresh_response.status_code != 200:
                raise Exception(
                    f"Token refresh failed ({refresh_response.status_code}): {refresh_response.text}"
                )

            refreshed_payload = refresh_response.json()
            if not refreshed_payload.get("access_token"):
                raise Exception("Token refresh response does not include access_token")

            merged_payload = dict(oauth_data)
            merged_payload.update(refreshed_payload)
            self._save_oauth_payload(merged_payload)
        except requests.exceptions.RequestException as e:
            raise Exception(f"Token refresh request failed: {str(e)}")
        except ValueError as e:
            raise Exception(f"Token refresh response is not valid JSON: {str(e)}")

    def _validate_return_size(self, return_value: str) -> str:
        f"""
        Check whether the return size is too large or not.
        If the return value exceeds {self.valves.MAX_RESPONSE_SIZE}, it will return an error message.
        """
        if len(return_value) > self.valves.MAX_RESPONSE_SIZE:
            return "Error: the response from Graph API is too large to be processed. Please limit the size of the page by using pagination parameters, or by filtering the results with specific conditions."
        return return_value

    def teams(
            self,
            filter_param: str = None,
            select: str = None,
            top: int = None,
            skiptoken: str = None,
            count: str = None,
    ):
        f"""
        List all teams in an organization.
        
        :param filter_param: The filter parameter in the Microsoft Graph API /teams endpoint allows you to narrow down the results of your request by returning only the teams that match specific criteria you define, using OData query syntax. Here are some examples:
                              - Find a team with a specific display name: https://graph.microsoft.com/v1.0/teams?$filter=displayName eq 'Project Apollo'
                              - Find teams that have a specific description: https://graph.microsoft.com/v1.0/teams?$filter=description eq 'Engineering Team'
                              - Find teams whose names start with a specific prefix: https://graph.microsoft.com/v1.0/teams?$filter=startswith(displayName, 'Eng')
        :param select: The select parameter in the Microsoft Graph API /teams endpoint allows you to limit the response to only the specific properties you actually need, rather than downloading the entire default payload. Here are some examples:
                        - Retrieve only the ID and display name of teams: https://graph.microsoft.com/v1.0/teams?$select=id,displayName
                        - Retrieve only the web URL of teams to create direct links: https://graph.microsoft.com/v1.0/teams?$select=webUrl
                        - Retrieve only the summary descriptions of teams: https://graph.microsoft.com/v1.0/teams?$select=description
                       Performance Tip: Using $select is a best practice for API efficiency. By turning off the data firehose and fetching only two or three properties, you significantly reduce network latency and memory usage in your application.
        :param top: The top parameter in the Microsoft Graph API /teams endpoint limits the size of the response by specifying the maximum number of items to return in a single page. Here are some examples:
                     - Retrieve only the top 5 teams: https://graph.microsoft.com/v1.0/teams?$top=5
                     - Retrieve only the top 10 teams: https://graph.microsoft.com/v1.0/teams?$top=10
                     - Retrieve a single team to test your connection payload: https://graph.microsoft.com/v1.0/teams?$top=1
                    Pagination Tip: If you have more teams in your tenant than the number you request in $top, Microsoft Graph will include a @odata.nextLink property in the JSON response body. This gives you a pre-formatted URL to fetch the next batch of teams, allowing you to cycle through your data safely without hitting performance limits.
        :param skiptoken: The skiptoken parameter is an opaque, stateful identifier used by Microsoft Graph to retrieve the next page of results when a dataset is too large to fit into a single response. Here are some examples:
                           - Fetch the second page of a specific paginated request: https://graph.microsoft.com/v1.0/teams?$skiptoken=X'4453707402000100000017'
                           - Fetch the next batch of results when dealing with heavy directory loads: https://graph.microsoft.com/v1.0/teams?$skiptoken=MSwwLDE1OTgwMzU4MTE4OTQ
                          Crucial Best Practice: Unlike $top or $select, you should never manually invent or guess a $skiptoken string. It is dynamically generated by the Microsoft Graph backend. When you request a large list of teams, Microsoft Graph will automatically append this exact parameter to the @odata.nextLink URL inside the JSON response. Your application should simply read that entire URL and feed it directly into the next HTTP request.
        :param count: The count parameter in the Microsoft Graph API /teams endpoint is used to retrieve the total number of items in a collection, either as an inline count included alongside the data or as a single integer value. Here are some examples:
                       - Retrieve the list of teams and include the total count in the metadata: https://graph.microsoft.com/v1.0/teams?$count=true
                       - Retrieve only the absolute total number of teams as a single integer: https://graph.microsoft.com/v1.0/teams/$count <- For this, please feed an empty string.
        :return: Collection of team objects in the response body.
        """
        url = "https://graph.microsoft.com/v1.0/teams"
        query_params = {}
        if filter_param: query_params["$filter"] = filter_param
        if select: query_params["$select"] = select
        if top: query_params["$top"] = top
        if skiptoken: query_params["$skiptoken"] = skiptoken
        if count: query_params["$count"] = count
        try:
            response = requests.get(url, headers=self._common_http_headers(), params=query_params, timeout=10)
            if self._is_token_expired_response(response):
                self._refresh_graph_access_token()
                response = requests.get(url, headers=self._common_http_headers(), params=query_params, timeout=10)
            if response.status_code == 200:
                return self._validate_return_size(response.text)
            else:
                return f"Error {response.status_code}: {response.reason} - {response.text}"
        except requests.exceptions.RequestException as e:
            return f"Connection Error: {str(e)}"

```
