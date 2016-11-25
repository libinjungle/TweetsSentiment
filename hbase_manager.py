import happybase

class HBase(object):
    table_name = "tweets_sentiment"
    row_count = 0
    namespace = "bl1810"
    batch_size = 1000
    host = "128.122.215.51"

    @staticmethod
    def connect_to_hbase():
        conn = happybase.Connection(host=HBase.host,
                                    table_prefix=HBase.namespace,
                                    table_prefix_separator=":")
        conn.open()
        table = conn.table(HBase.table_name)
        batch = table.batch(batch_size=HBase.batch_size)
        return conn, batch


    @staticmethod
    def insert_row(batch, row):
        batch.put(row[0]+':'+row[1], {"data:tweet": row[2], "data:sentiment":row[3]})
