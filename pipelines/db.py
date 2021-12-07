import psycopg2

from config import config

def connect():
    conn = None

    try:
        params = config()

        #connecting to pgsql server
        print('Connecting to pgsql server')
        conn = psycopg2.connect(**params)

        # create a cursor
        cur  = conn.cursor()
        
        # execute a statement 
        print('PgSQL db version')
        cur.execute('select version()')

        db_version = cur.fetchone()

        print(f"Db version here is {db_version}")
        cur.close()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    
    finally:
        if conn is not None:
            conn.close()
            print("DB connection is closed")

if __name__=="__main__":
    connect()