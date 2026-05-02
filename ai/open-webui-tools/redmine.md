## About

It's a tool for [Redmine API](https://www.redmine.org/projects/redmine/wiki/rest_api).

Supported Features:
* [Issues](https://www.redmine.org/projects/redmine/wiki/Rest_Issues)
* [Projects](https://www.redmine.org/projects/redmine/wiki/Rest_Projects)
* [Time Entries](https://www.redmine.org/projects/redmine/wiki/Rest_TimeEntries)
* [Users](https://www.redmine.org/projects/redmine/wiki/Rest_Users)
* [Wiki Pages](https://www.redmine.org/projects/redmine/wiki/Rest_WikiPages)

## Code

```py
from pydantic import BaseModel, Field
import requests
from typing import Dict
import json


class Tools:
    class Valves(BaseModel):
        REDMINE_BASE_URL: str = Field(
            default="https://redmine.example.com",
            description="The base URL of your Redmine server.",
        )
        REDMINE_API_KEY: str = Field(
            default="", description="Your Redmine API Access Key."
        )

    def __init__(self):
        self.valves = self.Valves()

    def _common_http_headers(self) -> Dict[str, str]:
        """
        Provides the HTTP headers commonly used to call Redmine APIs.

        :return: HTTP headers for calling Redmine APIs.
        """
        return {
            "X-Redmine-API-Key": self.valves.REDMINE_API_KEY,
            "Content-Type": "application/json",
        }

    def issues(
            self,
            offset: int = None,
            limit: int = None,
            sort: str = None,
            include: str = None,
            issue_id: str = None,
            project_id: int = None,
            subproject_id: str = None,
            tracker_id: int = None,
            status_id: str = None,
            assigned_to_id: str = None,
            parent_id: int = None,
    ) -> str:
        f"""
        Fetches a list of issues from Redmine and returns a raw JSON string or an error message.
        It calls the `GET {self.valves.REDMINE_BASE_URL.rstrip('/')}/issues.json` endpoint.

        :param offset: the offset of the first object to retrieve.
        :param limit: the number of items to be present in the response (default is 25, maximum is 100).
        :param sort: column to sort with. Append :desc to invert the order.
        :param include: fetch associated data (optional, use comma to fetch multiple associations). Possible values:
                         * attachments - Since 3.4.0
                         * relations
        :param issue_id: get issue with the given id or multiple issues by id using ',' to separate id.
        :param project_id: get issues from the project with the given id (a numeric value, not a project identifier).
        :param subproject_id: get issues from the subproject with the given id. You can use project_id=XXX&subproject_id=!* to get only the issues of a given project and none of its subprojects.
        :param tracker_id: get issues from the tracker with the given id.
        :param status_id: get issues with the given status id only. Possible values: open, closed, * to get open and closed issues, status id.
        :param assigned_to_id: get issues which are assigned to the given user id. me can be used instead an ID to fetch all issues from the logged in user (via API key or HTTP auth).
        :param parent_id: get issues whose parent issue is given id.

        :return: Redmine issues.
        """
        url = f"{self.valves.REDMINE_BASE_URL.rstrip('/')}/issues.json"
        query_params = {}
        if offset: query_params["offset"] = offset
        if limit: query_params["limit"] = limit
        if sort: query_params["sort"] = sort
        if include: query_params["include"] = include
        if issue_id: query_params["issue_id"] = issue_id
        if project_id: query_params["project_id"] = project_id
        if subproject_id: query_params["subproject_id"] = subproject_id
        if tracker_id: query_params["tracker_id"] = tracker_id
        if status_id: query_params["status_id"] = status_id
        if assigned_to_id: query_params["assigned_to_id"] = assigned_to_id
        if parent_id: query_params["parent_id"] = parent_id
        try:
            response = requests.get(url, headers=self._common_http_headers(), params=query_params)
            if response.status_code == 200:
                return response.text
            else:
                return f"Error {response.status_code}: {response.reason} - {response.text}"
        except requests.exceptions.RequestException as e:
            return f"Connection Error: {str(e)}"

    def issue(
            self,
            issue_id: int,
            include: str = None,
            journals_offset: int = 0,
            journals_limit: int = 25,
    ) -> str:
        f"""
        Fetches the information of the target issue from Redmine and returns a raw JSON string or an error message.
        It calls the `GET {self.valves.REDMINE_BASE_URL.rstrip('/')}/issues/{issue_id}.json` endpoint.
        The actual API does not support journal pagination.
        Since there might be too many journals, it mimics Redmine-style pagination from other endpoints.

        :param issue_id: the ID of the target issue.
        :param include: fetch associated data (optional, use comma to fetch multiple associations). Possible values:
                * children
                * attachments
                * relations
                * changesets
                * journals
                * watchers - Since 2.3.0
                * allowed_statuses - Since 5.0.x, Returns the available allowed statuses (the same values as provided in the issue edit form) based on:
                    * the issue's current tracker, the issue's current status, and the member's role (the defined workflow);
                    * the existence of any open subtask(s);
                    * the existence of any open blocking issue(s);
                    * the existence of a closed parent issue.
        :param journals_offset: the offset of the first journal to retrieve when fetching journals with include=journal.
        :param journals_limit: the number of journals to retrieve when fetching journals with include=journal. The default is 25, and maximum is 100.

        :return: Redmine issue.
        """
        url = f"{self.valves.REDMINE_BASE_URL.rstrip('/')}/issues/{issue_id}.json"
        query_params = {}
        if include: query_params["include"] = include
        try:
            response = requests.get(url, headers=self._common_http_headers(), params=query_params)
            if response.status_code == 200:
                response_json = response.json()
                if "issue" in response_json and "journals" in response_json["issue"]:
                    effective_limit = min(journals_limit, 100)
                    all_journals = response_json["issue"]["journals"]
                    total_journals = len(all_journals)
                    paginated_journals = all_journals[journals_offset: journals_offset + effective_limit]
                    response_json["issue"]["journals"] = paginated_journals
                    response_json["issue"]["journals_total_count"] = total_journals
                    response_json["issue"]["journals_offset"] = journals_offset
                    response_json["issue"]["journals_limit"] = effective_limit
                return json.dumps(response_json)
            else:
                return f"Error {response.status_code}: {response.reason} - {response.text}"
        except requests.exceptions.RequestException as e:
            return f"Connection Error: {str(e)}"

    def projects(
            self,
            offset: int = None,
            limit: int = None,
            include: str = None,
    ) -> str:
        f"""
        Fetches projects from Redmine and returns a raw JSON string or an error message.
        It calls the `GET {self.valves.REDMINE_BASE_URL.rstrip('/')}/projects.json` endpoint.

        :param offset: the offset of the first object to retrieve.
        :param limit: the number of items to be present in the response (default is 25, maximum is 100).
        :param include: fetch associated data (optional). Values should be separated by a comma ",". Possible values:
                * trackers
                * issue_categories
                * enabled_modules (since 2.6.0)
                * time_entry_activities (since 3.4.0)
                * issue_custom_fields (since 4.2.0)

        :return: Redmine projects.
        """
        url = f"{self.valves.REDMINE_BASE_URL.rstrip('/')}/projects.json"
        query_params = {}
        if offset: query_params["offset"] = offset
        if limit: query_params["limit"] = limit
        if include: query_params["include"] = include
        try:
            response = requests.get(url, headers=self._common_http_headers(), params=query_params)
            if response.status_code == 200:
                return response.text
            else:
                return f"Error {response.status_code}: {response.reason} - {response.text}"
        except requests.exceptions.RequestException as e:
            return f"Connection Error: {str(e)}"

    def time_entries(
            self,
            offset: int = None,
            limit: int = None,
            user_id: int = None,
            project_id: str = None,
            spent_from: str = None,
            spent_to: str = None,
    ) -> str:
        f"""
        Fetches time entries from Redmine and returns a raw JSON string or an error message.
        It calls the `GET {self.valves.REDMINE_BASE_URL.rstrip('/')}/time_entries.json` endpoint.

        :param offset: the offset of the first object to retrieve.
        :param limit: the number of items to be present in the response (default is 25, maximum is 100).
        :param user_id: get time entries by the given user id.
        :param project_id: when filtering by project id, you can use either project numeric ID or its string identifier.
        :param spent_from: get time entries from that date (inclusive). Format example: yyyy-mm-dd.
        :param spent_to: get time entries until that date (inclusive). Format example: yyyy-mm-dd.

        :return: Redmine time entries.
        """
        url = f"{self.valves.REDMINE_BASE_URL.rstrip('/')}/time_entries.json"
        query_params = {}
        if offset: query_params["offset"] = offset
        if limit: query_params["limit"] = limit
        if user_id: query_params["user_id"] = user_id
        if project_id: query_params["project_id"] = project_id
        if spent_from: query_params["from"] = spent_from
        if spent_to: query_params["to"] = spent_to
        try:
            response = requests.get(url, headers=self._common_http_headers(), params=query_params)
            if response.status_code == 200:
                return response.text
            else:
                return f"Error {response.status_code}: {response.reason} - {response.text}"
        except requests.exceptions.RequestException as e:
            return f"Connection Error: {str(e)}"

    def users(
            self,
            offset: int = None,
            limit: int = None,
            status: int = None,
            name: str = None,
            group_id: int = None,
    ) -> str:
        f"""
        Fetches users from Redmine and returns a raw JSON string or an error message.
        It calls the `GET {self.valves.REDMINE_BASE_URL.rstrip('/')}/users.json` endpoint.
        This endpoint requires admin privileges.

        :param offset: the offset of the first object to retrieve.
        :param limit: the number of items to be present in the response (default is 25, maximum is 100).
        :param status: get only users with the given status. See app/models/principal.rb for a list of available statuses. Supply an empty value to match all users regardless of their status. Default is 1 (active users). Possible values are:
                        * 1: Active (User can login and use their account)
                        * 2: Registered (User has registered but not yet confirmed their email address or was not yet activated by an administrator. User can not login)
                        * 3: Locked (User was once active and is now locked, User can not login)
        :param name: filter users on their login, firstname, lastname and mail ; if the pattern contains a space, it will also return users whose firstname match the first word or lastname match the second word.
        :param group_id: get only users who are members of the given group.

        :return: Redmine users.
        """
        url = f"{self.valves.REDMINE_BASE_URL.rstrip('/')}/users.json"
        query_params = {}
        if offset: query_params["offset"] = offset
        if limit: query_params["limit"] = limit
        if status: query_params["status"] = status
        if name: query_params["name"] = name
        if group_id: query_params["group_id"] = group_id
        try:
            response = requests.get(url, headers=self._common_http_headers(), params=query_params)
            if response.status_code == 200:
                return response.text
            else:
                return f"Error {response.status_code}: {response.reason} - {response.text}"
        except requests.exceptions.RequestException as e:
            return f"Connection Error: {str(e)}"

    def wikis(
            self,
            project_identifier: str,
            offset: int = 0,
            limit: int = 25,
    ) -> str:
        f"""
        Fetches the list of wiki pages from Redmine and returns a raw JSON string or an error message.
        It calls the `GET {self.valves.REDMINE_BASE_URL.rstrip('/')}/projects/{project_identifier}/wiki/index.json` endpoint.
        The actual API does not support pagination.
        Since there might be too many wiki pages, it mimics Redmine-style pagination from other endpoints.

        :param project_identifier: you can see this value from calling `/projects` endpoint. Each project has its identifier.
        :param offset: the offset of the first object to retrieve.
        :param limit: the number of items to be present in the response (default is 25, maximum is 100).

        :return: Redmine wiki pages.
        """
        url = f"{self.valves.REDMINE_BASE_URL.rstrip('/')}/projects/{project_identifier}/wiki/index.json"
        effective_limit = min(limit, 100)
        try:
            response = requests.get(url, headers=self._common_http_headers())
            if response.status_code == 200:
                all_pages = response.json().get("wiki_pages", [])
                total_count = len(all_pages)
                paginated_pages = all_pages[offset: offset + effective_limit]
                result = {
                    "wiki_pages": paginated_pages,
                    "total_count": total_count,
                    "offset": offset,
                    "limit": effective_limit
                }
                return json.dumps(result)
            else:
                return f"Error {response.status_code}: {response.reason} - {response.text}"
        except requests.exceptions.RequestException as e:
            return f"Connection Error: {str(e)}"

    def wiki(
            self,
            project_identifier: str,
            title: str,
            include: str = None,
    ) -> str:
        f"""
        Fetches the wiki page from Redmine and returns a raw JSON string or an error message.
        It calls the `GET {self.valves.REDMINE_BASE_URL.rstrip('/')}/projects/{project_identifier}/wiki/{title}.json` endpoint.

        :param project_identifier: you can see this value from calling `/projects` endpoint. Each project has its identifier.
        :param title: you cann see this value from calling `/projects/{project_identifier}/wiki/index.json` endpoint.
        :param include: fetch associated data (optional, use comma to fetch multiple associations). Possible values:
                         * attachments - Since 3.4.0

        :return: Redmine wiki page.
        """
        url = f"{self.valves.REDMINE_BASE_URL.rstrip('/')}/projects/{project_identifier}/wiki/{title}.json"
        query_params = {}
        if include: query_params["include"] = include
        try:
            response = requests.get(url, headers=self._common_http_headers(), params=query_params)
            if response.status_code == 200:
                return response.text
            else:
                return f"Error {response.status_code}: {response.reason} - {response.text}"
        except requests.exceptions.RequestException as e:
            return f"Connection Error: {str(e)}"

    def old_wiki(
            self,
            project_identifier: str,
            title: str,
            version: int,
            include: str = None,
    ) -> str:
        f"""
        Fetches the old version of the wiki page from Redmine and returns a raw JSON string or an error message.
        It calls the `GET {self.valves.REDMINE_BASE_URL.rstrip('/')}/projects/{project_identifier}/wiki/{title}/{version}.json` endpoint.

        :param project_identifier: you can see this value from calling `/projects` endpoint. Each project has its identifier.
        :param title: you cann see this value from calling `/projects/{project_identifier}/wiki/index.json` endpoint.
        :param version: the version of the wiki.
        :param include: fetch associated data (optional, use comma to fetch multiple associations). Possible values:
                         * attachments - Since 3.4.0

        :return: The old version of the Redmine wiki page.
        """
        url = f"{self.valves.REDMINE_BASE_URL.rstrip('/')}/projects/{project_identifier}/wiki/{title}/{version}.json"
        query_params = {}
        if include: query_params["include"] = include
        try:
            response = requests.get(url, headers=self._common_http_headers(), params=query_params)
            if response.status_code == 200:
                return response.text
            else:
                return f"Error {response.status_code}: {response.reason} - {response.text}"
        except requests.exceptions.RequestException as e:
            return f"Connection Error: {str(e)}"
```
