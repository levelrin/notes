## About

We want to call [Microsoft Graph API](https://learn.microsoft.com/en-us/graph/overview).

## Supported Features

The following Graph API endpoints are supported:
 - [List teams](https://learn.microsoft.com/en-us/graph/api/teams-list?view=graph-rest-1.0&tabs=http)
 - [List joinedTeams](https://learn.microsoft.com/en-us/graph/api/user-list-joinedteams?view=graph-rest-1.0&tabs=http)
 - [Get team](https://learn.microsoft.com/en-us/graph/api/team-get?view=graph-rest-1.0&tabs=http)
 - [List allChannels](https://learn.microsoft.com/en-us/graph/api/team-list-allchannels?view=graph-rest-1.0&tabs=http)
 - [List channels](https://learn.microsoft.com/en-us/graph/api/channel-list?view=graph-rest-1.0&tabs=http)
 - [Get channel](https://learn.microsoft.com/en-us/graph/api/channel-get?view=graph-rest-1.0&tabs=http)
 - [Get primaryChannel](https://learn.microsoft.com/en-us/graph/api/team-get-primarychannel?view=graph-rest-1.0&tabs=http)
 - [List channel messages](https://learn.microsoft.com/en-us/graph/api/channel-list-messages?view=graph-rest-1.0&tabs=http)

## Setup

You should bring our own token as a JSON file.

Let's say the token is stored in this path: /app/backend/data/graph-oauth.json

We recommend adjusting the permission like this:
```sh
chmod o+rw /app/backend/data/graph-oauth.json
```

You are set!

Fortunately, this operation is one-time only.

Once you create the access token file, the Python code in the section below will use the refresh token and update the file automatically.

When you apply the Python code in the section below, you should configure valves, especially `OAUTH_JSON_PATH`.

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
            default=67000,
            description="The response from the Graph API can be too large for the model. This valve sets the maximum response size (in characters) that will be returned to the model. If the API response exceeds this size, it will return an error message suggesting limiting the size of the page."
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
            if response.status_code == 401:
                self._refresh_graph_access_token()
                response = requests.get(url, headers=self._common_http_headers(), params=query_params, timeout=10)
            if response.status_code == 200:
                return self._validate_return_size(response.text)
            else:
                return f"Error {response.status_code}: {response.reason} - {response.text}"
        except requests.exceptions.RequestException as e:
            return f"Connection Error: {str(e)}"

    def all_channels(
            self,
            team_id: str,
            filter_param: str = None,
            select: str = None,
    ):
        f"""
        Get the list of channels either in this team or shared with this team (incoming channels).
        Populating the email and moderationSettings properties for a channel is an expensive operation that results in slow performance. Use $select to exclude the email and moderationSettings properties to improve performance.

        :param team_id: The team ID.
        :param filter_param: The $filter parameter restricts the response to only return channels that match your specified criteria, such as checking a channel's name, membership type, or description. Here are some examples:
                              - Find a channel with a specific name: https://graph.microsoft.com/v1.0/teams/{team_id}/allChannels?$filter=displayName eq 'General'
                              - Find only the private channels within the team: https://graph.microsoft.com/v1.0/teams/{team_id}/allChannels?$filter=membershipType eq 'private'
                              - Find channels whose descriptions match a specific function: https://graph.microsoft.com/v1.0/teams/{team_id}/allChannels?$filter=description eq 'Regional Sales Coordination'
        :param select: The $select parameter limits the returned properties for each channel to just the fields you explicitly request, which prevents the API from wasting processing time on data you don't need. Here are some examples:
                        - Retrieve only the unique ID and display name of the channels: https://graph.microsoft.com/v1.0/teams/{team_id}/allChannels?$select=id,displayName
                        - Retrieve only the privacy/membership classification of the channels: https://graph.microsoft.com/v1.0/teams/{team_id}/allChannels?$select=membershipType
                        - Retrieve only the web URLs to link directly to the channels: https://graph.microsoft.com/v1.0/teams/{team_id}/allChannels?$select=webUrl
        :return: Collection of channel objects in the response body.
                 The response also includes the @odata.id property which can be used to access the channel and run other operations on the channel object.
                 When the result set spans multiple pages, the response includes an @odata.nextLink property with a URL for retrieving the next page of results. 
        """
        if not team_id or not str(team_id).strip():
            return "Invalid team_id: team_id must be a non-empty string"
        url = f"https://graph.microsoft.com/v1.0/teams/{team_id}/allChannels"
        query_params = {}
        if filter_param:
            query_params["$filter"] = filter_param
        if select:
            query_params["$select"] = select
        try:
            response = requests.get(
                url, headers=self._common_http_headers(), params=query_params, timeout=10
            )
            if response.status_code == 401:
                self._refresh_graph_access_token()
                response = requests.get(
                    url, headers=self._common_http_headers(), params=query_params, timeout=10
                )
            if response.status_code == 200:
                return self._validate_return_size(response.text)
            return f"Error {response.status_code}: {response.reason} - {response.text}"
        except requests.exceptions.RequestException as e:
            return f"Connection Error: {str(e)}"

    def joined_teams(self):
        """
        Get the teams that the current user is a member of.

        :return: Collection of team objects in the response body.
        """
        url = "https://graph.microsoft.com/v1.0/me/joinedTeams"
        try:
            response = requests.get(
                url, headers=self._common_http_headers(), timeout=10
            )
            if response.status_code == 401:
                self._refresh_graph_access_token()
                response = requests.get(
                    url, headers=self._common_http_headers(), timeout=10
                )
            if response.status_code == 200:
                return self._validate_return_size(response.text)
            return f"Error {response.status_code}: {response.reason} - {response.text}"
        except requests.exceptions.RequestException as e:
            return f"Connection Error: {str(e)}"

    def channels(
            self,
            team_id: str,
            filter_param: str = None,
            select: str = None,
    ):
        f"""
        Retrieve the list of channels in this team.

        :param team_id: The team ID.
        :param filter_param: The $filter parameter in the Microsoft Graph API /teams/{team_id}/channels endpoint allows you to narrow down the results by returning only the channels that match specific criteria. Here are some examples:
                              - Find a channel with a specific name: https://graph.microsoft.com/v1.0/teams/{team_id}/channels?$filter=displayName eq 'General'
                              - Find only standard channels: https://graph.microsoft.com/v1.0/teams/{team_id}/channels?$filter=membershipType eq 'standard'
                              - Find channels with a specific description: https://graph.microsoft.com/v1.0/teams/{team_id}/channels?$filter=description eq 'Regional Sales Coordination'
        :param select: The $select parameter in the Microsoft Graph API /teams/{team_id}/channels endpoint allows you to limit the response to only the specific properties you need from each channel object. Here are some examples:
                        - Retrieve only the unique ID and display name of channels: https://graph.microsoft.com/v1.0/teams/{team_id}/channels?$select=id,displayName
                        - Retrieve only the channel membership type: https://graph.microsoft.com/v1.0/teams/{team_id}/channels?$select=membershipType
                        - Retrieve only the web URLs for channels: https://graph.microsoft.com/v1.0/teams/{team_id}/channels?$select=webUrl
        :return: Collection of Channel objects in the response body.
        """
        if not team_id or not str(team_id).strip():
            return "Invalid team_id: team_id must be a non-empty string"
        url = f"https://graph.microsoft.com/v1.0/teams/{team_id}/channels"
        query_params = {}
        if filter_param:
            query_params["$filter"] = filter_param
        if select:
            query_params["$select"] = select
        try:
            response = requests.get(
                url, headers=self._common_http_headers(), params=query_params, timeout=10
            )
            if response.status_code == 401:
                self._refresh_graph_access_token()
                response = requests.get(
                    url, headers=self._common_http_headers(), params=query_params, timeout=10
                )
            if response.status_code == 200:
                return self._validate_return_size(response.text)
            return f"Error {response.status_code}: {response.reason} - {response.text}"
        except requests.exceptions.RequestException as e:
            return f"Connection Error: {str(e)}"

    def channel(
            self,
            team_id: str,
            channel_id: str,
            filter_param: str = None,
            select: str = None,
    ):
        f"""
        Retrieve the properties and relationships of a channel.
        This method supports federation. Only a user who is a member of the shared channel can retrieve channel information.

        :param team_id: The team ID.
        :param channel_id: The channel ID.
        :param filter_param: The $filter parameter in the Microsoft Graph API /teams/{team_id}/channels/{channel_id} endpoint allows you to narrow down the response by returning only channel data that matches specific criteria. Here are some examples:
                              - Find a channel with a specific name: https://graph.microsoft.com/v1.0/teams/{team_id}/channels/{channel_id}?$filter=displayName eq 'General'
                              - Find a channel by membership type: https://graph.microsoft.com/v1.0/teams/{team_id}/channels/{channel_id}?$filter=membershipType eq 'private'
                              - Find a channel with a specific description: https://graph.microsoft.com/v1.0/teams/{team_id}/channels/{channel_id}?$filter=description eq 'Regional Sales Coordination'
        :param select: The $select parameter in the Microsoft Graph API /teams/{team_id}/channels/{channel_id} endpoint allows you to limit the response to only the specific properties you need from the channel object. Here are some examples:
                        - Retrieve only the unique ID and display name of the channel: https://graph.microsoft.com/v1.0/teams/{team_id}/channels/{channel_id}?$select=id,displayName
                        - Retrieve only the channel membership type: https://graph.microsoft.com/v1.0/teams/{team_id}/channels/{channel_id}?$select=membershipType
                        - Retrieve only the web URL of the channel: https://graph.microsoft.com/v1.0/teams/{team_id}/channels/{channel_id}?$select=webUrl
        :return: A channel object in the response body.
        """
        if not team_id or not str(team_id).strip():
            return "Invalid team_id: team_id must be a non-empty string"
        if not channel_id or not str(channel_id).strip():
            return "Invalid channel_id: channel_id must be a non-empty string"
        url = f"https://graph.microsoft.com/v1.0/teams/{team_id}/channels/{channel_id}"
        query_params = {}
        if filter_param:
            query_params["$filter"] = filter_param
        if select:
            query_params["$select"] = select
        try:
            response = requests.get(
                url, headers=self._common_http_headers(), params=query_params, timeout=10
            )
            if response.status_code == 401:
                self._refresh_graph_access_token()
                response = requests.get(
                    url, headers=self._common_http_headers(), params=query_params, timeout=10
                )
            if response.status_code == 200:
                return self._validate_return_size(response.text)
            return f"Error {response.status_code}: {response.reason} - {response.text}"
        except requests.exceptions.RequestException as e:
            return f"Connection Error: {str(e)}"

    def team(
            self,
            team_id: str,
            select: str = None,
            expand: str = None,
    ):
        f"""
        Retrieve the properties and relationships of the specified team.

        :param team_id: The team ID.
        :param select: The $select parameter in the Microsoft Graph API /teams/{team_id} endpoint allows you to limit the response to only the specific properties you need from the team object. Here are some examples:
                        - Retrieve only the ID and display name of the team: https://graph.microsoft.com/v1.0/teams/{team_id}?$select=id,displayName
                        - Retrieve only the team description: https://graph.microsoft.com/v1.0/teams/{team_id}?$select=description
                        - Retrieve only the web URL of the team: https://graph.microsoft.com/v1.0/teams/{team_id}?$select=webUrl
        :param expand: The $expand parameter in the Microsoft Graph API /teams/{team_id} endpoint allows you to include related resources inline with the team object in a single request. Here are some examples:
                        - Retrieve the team and expand installed apps: https://graph.microsoft.com/v1.0/teams/{team_id}?$expand=installedApps
                        - Retrieve the team and expand channels: https://graph.microsoft.com/v1.0/teams/{team_id}?$expand=channels
                        - Retrieve the team and expand members: https://graph.microsoft.com/v1.0/teams/{team_id}?$expand=members
        :return: A team object in the response body.
        """
        if not team_id or not str(team_id).strip():
            return "Invalid team_id: team_id must be a non-empty string"
        url = f"https://graph.microsoft.com/v1.0/teams/{team_id}"
        query_params = {}
        if select:
            query_params["$select"] = select
        if expand:
            query_params["$expand"] = expand
        try:
            response = requests.get(
                url, headers=self._common_http_headers(), params=query_params, timeout=10
            )
            if response.status_code == 401:
                self._refresh_graph_access_token()
                response = requests.get(
                    url, headers=self._common_http_headers(), params=query_params, timeout=10
                )
            if response.status_code == 200:
                return self._validate_return_size(response.text)
            return f"Error {response.status_code}: {response.reason} - {response.text}"
        except requests.exceptions.RequestException as e:
            return f"Connection Error: {str(e)}"

    def primary_channel(
            self,
            team_id: str,
            filter_param: str = None,
            select: str = None,
            expand: str = None,
    ):
        f"""
        Get the default channel, General, of a team.

        :param team_id: The team ID.
        :param filter_param: The $filter parameter in the Microsoft Graph API /teams/{team_id}/primaryChannel endpoint allows you to customize the response by applying OData filter expressions. Here are some examples:
                              - Filter by display name: https://graph.microsoft.com/v1.0/teams/{team_id}/primaryChannel?$filter=displayName eq 'General'
                              - Filter by membership type: https://graph.microsoft.com/v1.0/teams/{team_id}/primaryChannel?$filter=membershipType eq 'standard'
                              - Filter by description: https://graph.microsoft.com/v1.0/teams/{team_id}/primaryChannel?$filter=description eq 'General discussions'
        :param select: The $select parameter in the Microsoft Graph API /teams/{team_id}/primaryChannel endpoint allows you to limit the response to only the specific properties you need from the channel object. Here are some examples:
                        - Retrieve only the unique ID and display name of the channel: https://graph.microsoft.com/v1.0/teams/{team_id}/primaryChannel?$select=id,displayName
                        - Retrieve only the channel membership type: https://graph.microsoft.com/v1.0/teams/{team_id}/primaryChannel?$select=membershipType
                        - Retrieve only the web URL of the channel: https://graph.microsoft.com/v1.0/teams/{team_id}/primaryChannel?$select=webUrl
        :param expand: The $expand parameter in the Microsoft Graph API /teams/{team_id}/primaryChannel endpoint allows you to include related resources inline with the channel object in a single request. Here are some examples:
                        - Retrieve the channel and expand tabs: https://graph.microsoft.com/v1.0/teams/{team_id}/primaryChannel?$expand=tabs
                        - Retrieve the channel and expand members: https://graph.microsoft.com/v1.0/teams/{team_id}/primaryChannel?$expand=members
                        - Retrieve the channel and expand filesFolder: https://graph.microsoft.com/v1.0/teams/{team_id}/primaryChannel?$expand=filesFolder
        :return: A channel object in the response body.
        """
        if not team_id or not str(team_id).strip():
            return "Invalid team_id: team_id must be a non-empty string"
        url = f"https://graph.microsoft.com/v1.0/teams/{team_id}/primaryChannel"
        query_params = {}
        if filter_param:
            query_params["$filter"] = filter_param
        if select:
            query_params["$select"] = select
        if expand:
            query_params["$expand"] = expand
        try:
            response = requests.get(
                url, headers=self._common_http_headers(), params=query_params, timeout=10
            )
            if response.status_code == 401:
                self._refresh_graph_access_token()
                response = requests.get(
                    url, headers=self._common_http_headers(), params=query_params, timeout=10
                )
            if response.status_code == 200:
                return self._validate_return_size(response.text)
            return f"Error {response.status_code}: {response.reason} - {response.text}"
        except requests.exceptions.RequestException as e:
            return f"Connection Error: {str(e)}"

    def channel_messages(
            self,
            team_id: str,
            channel_id: str,
            top: int = None,
            expand: str = None,
    ):
        f"""
        Retrieve the list of messages (without the replies) in a channel of a team.
        To get the replies for a message, call the list message replies or the get message reply API.
        This method supports federation. To list channel messages in application context, the request must be made from the tenant that the channel owner belongs to (represented by the tenantId property on the channel).

        :param team_id: The team ID.
        :param channel_id: The channel ID.
        :param top: Apply $top to specify the number of channel messages returned per page in the response. The default page size is 20 messages. You can extend up to 50 channel messages per page. Here are some examples:
                     - Retrieve only the top 10 messages: https://graph.microsoft.com/v1.0/teams/{team_id}/channels/{channel_id}/messages?$top=10
                     - Retrieve only the top 20 messages: https://graph.microsoft.com/v1.0/teams/{team_id}/channels/{channel_id}/messages?$top=20
                     - Retrieve the maximum 50 messages per page: https://graph.microsoft.com/v1.0/teams/{team_id}/channels/{channel_id}/messages?$top=50
        :param expand: Apply $expand to get the properties of channel messages that are replies. By default, a response can include up to 200 replies. For an operation that expands channel messages with more than 200 replies, use the request URL returned in replies@odata.nextLink to get the next page of replies. Here are some examples:
                        - Expand replies in each root message: https://graph.microsoft.com/v1.0/teams/{team_id}/channels/{channel_id}/messages?$expand=replies
                        - Expand replies with selected fields: https://graph.microsoft.com/v1.0/teams/{team_id}/channels/{channel_id}/messages?$expand=replies($select=id,from,body)
                        - Expand replies and include the sender info: https://graph.microsoft.com/v1.0/teams/{team_id}/channels/{channel_id}/messages?$expand=replies($expand=from)
        :return: A collection of chatMessage objects in the response body. The channel messages in the response are sorted by the last modified date of the entire reply chain, including both the root channel message and its replies.
        """
        if not team_id or not str(team_id).strip():
            return "Invalid team_id: team_id must be a non-empty string"
        if not channel_id or not str(channel_id).strip():
            return "Invalid channel_id: channel_id must be a non-empty string"
        if top is not None and (top < 1 or top > 50):
            return "Invalid top: top must be between 1 and 50"
        url = f"https://graph.microsoft.com/v1.0/teams/{team_id}/channels/{channel_id}/messages"
        query_params = {}
        if top is not None:
            query_params["$top"] = top
        if expand:
            query_params["$expand"] = expand
        try:
            response = requests.get(
                url, headers=self._common_http_headers(), params=query_params, timeout=10
            )
            if response.status_code == 401:
                self._refresh_graph_access_token()
                response = requests.get(
                    url, headers=self._common_http_headers(), params=query_params, timeout=10
                )
            if response.status_code == 200:
                return self._validate_return_size(response.text)
            return f"Error {response.status_code}: {response.reason} - {response.text}"
        except requests.exceptions.RequestException as e:
            return f"Connection Error: {str(e)}"

```
