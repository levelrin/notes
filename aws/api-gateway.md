## Enable Log

Since log is off by default, we need to enable it.

We need to create a role if we have not:
1. Go to `IAM`.
2. Go to `Roles`.
3. Click `Create role`.
4. Make sure `AWS service` is selected.
5. Find `API Gateway` under `Use cases for other AWS services:`, select `API Gateway`, and click `Next`.
6. Click `Next` on `Add permissions` page.
7. Put `Role name` and click `Create role`.

After that, we can enable log like this:
1. Select the API.
2. Go to `Settings` below `Client Certificates`.
3. Put the `CloudWatch log role ARN*` and click `Save`.
    - It's the ARN of the role we created above.
4. Go to `Stages` and select the stage that you want to enable log.
5. Go to `Logs/Tracing` tab.
6. Select the log level that you want from `CloudWatch Logs`.
7. Click `Save Changes`.

The log should be available in `CloudWatch` from now.

## Return static page or data without actual integration

Steps for `REST API`:
1. Create a method if you have not:
    1. Go to `Resources`.
    2. Click `Actions` and `Create Method`.
2. Select the method.
3. Select the `Mock` for `Integration type` and click `Save`.
4. Click `Integration Response` and expand the table.
5. Expand `Mapping Templates`.
6. Click `Add mapping template`.
7. Put `Content-Type` (ex: text/html) and click the checkmark.
8. A text area will appear on the right. Put the value you want, which will be returned by the endpoint.
9. Click `Save`.
10. Deploy.
    1. Click `Actions` and `Deploy API`.

Note that the modification of return value may not be reflected immediately after deploy.
It takes some time.
