
### . Run the API locally

```shell
set -a && source .env.example && set +a
uvicorn main:app --host 0.0.0.0 --port 8081
```

Open your browser and access this address `localhost:8081/docs` to access API doc (Swagger UI).

### 3. Enjoy your API

There are two routes in the Swagger UI:

- `/chat`: Press `Try it out`, then enter `Show me the image url of Saint Laurent YSL Men's Black/Brown Leather Zip Around Wallet` in the `text` and press `Execute`
