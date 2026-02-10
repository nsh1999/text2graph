from db import PostgreSQLConnection


def main():
    print("Hello from text2graph!")
    pg = PostgreSQLConnection(
        min_connections=1,
        max_connections=5,
    )


if __name__ == "__main__":
    main()
