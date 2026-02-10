from db import PostgreSQLConnection


def main():
    print("Hello from text2graph!")
    pg = PostgreSQLConnection(
        #host="localhost",
        #export=5432,
        #database="capybara_mac",
        #user="arxiv",
        #password="BRA>ket42",
        min_connections=1,
        max_connections=5,
    )


if __name__ == "__main__":
    main()
