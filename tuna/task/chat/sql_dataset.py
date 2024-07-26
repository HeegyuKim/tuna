import json

from .datasets import ChatDataSource, BaseAlpacaDataSource, datasources, DatasetArguments, VicunaChatDataSource
from datasets import load_dataset, Dataset
from copy import deepcopy


TEXT2SQL_SYSTEM_PROMPT = """Your task is to convert natural language questions into valid SQL queries based on the provided database schema. Follow these guidelines strictly:

1. Analyze the given database schema carefully. Only use tables and columns that are explicitly defined in the schema.
2. Use only SQL syntax and functions that are supported by SQLite3. Avoid using advanced features or syntax that SQLite3 doesn't support.
3. Do not write comments in SQL query. Only write SQL query.

Your response should be a following format:
```sql
...
```"""

TEXT2SQL_USER_PROMPT = """
**Schema**
{schema}

**Question**
{question}"""

TEXT2SQL_USER_HINT_PROMPT = """
**Schema**
{schema}

**Question**
{question}

**Hint**
{hint}"""

TABLE_PROMPT = """{schema}
- Number of rows: {num_rows}"""

def table2text(table_dict, add_statistics=True):
    table_name = table_dict['table_name']
    table_text = TABLE_PROMPT.format(schema=table_dict['schema'], num_rows=table_dict['num_rows'])
    columns = json.loads(table_dict['columns'])

    if columns and add_statistics:
        table_text += f"\n**Statistics for {table_name}**\nNumeric columns:\n"
        for column in columns:
            if "statistics" in column:
                stats = column['statistics']
                table_text += f"- {column['column_name']}: min={stats['min']}, max={stats['max']}, mean={stats['mean']}, std={stats['std']}, 25%={stats['25%']}, 50%={stats['50%']}, 75%={stats['75%']}\n"
            else:
                topk = column['top_value_counts']
                topk_str = ", ".join([f"{k} ({v})" if len(k) <= 20 else f"{k[:20]} ... {len(k)} chars ({v})" for k, v in topk.items()])
                k = len(topk)

                table_text += f"- {column['column_name']}: {column['unique_values']} unique values, top {k} (frequency): {topk_str}\n"

    return table_text

def schema2text(schema_dict, add_statistics=True):
    tables = [table2text(table, add_statistics=add_statistics) for table in schema_dict['tables']]
    return f"Database: {schema_dict['db_id']}\n" + "\n\n".join(tables)


class BaseSQLDataSource(ChatDataSource):
    SCHEMA_WITH_STATS = False

    def get_schema_dataset(self, split: str):
        raise NotImplementedError

    def get_sql_dataset(self, args: DatasetArguments, split: str):
        raise NotImplementedError

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:

        schema = self.get_schema_dataset(split)
        schema_dict = {}
        for item in schema:
            schema_dict[item['db_id']] = schema2text(item, self.SCHEMA_WITH_STATS)
        
        ds = self.get_sql_dataset(args, split)

        def map_conversations(item):
            db_id = item["db_id"]
            hint = item.get("evidence")
            question = item["question"]
            query = item["query"]

            schema_text = schema_dict[db_id]
            if hint:
                user_prompt = TEXT2SQL_USER_PROMPT.format(schema=schema_text, question=question)
            else:
                user_prompt = TEXT2SQL_USER_HINT_PROMPT.format(schema=schema_text, question=question, hint=hint)

            return {
                "conversations": [
                    {"role": "system", "content": TEXT2SQL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": query}
                ]
            }

        ds = ds.map(map_conversations)
        return ds

@datasources("xlangai/spider")
class SpiderInstruct(BaseSQLDataSource):
    SCHEMA_WITH_STATS = False

    def get_schema_dataset(self, split: str):
        return load_dataset("iknow-lab/spider-schema", split="train")

    def get_sql_dataset(self, args: DatasetArguments, split: str):
        if split == "test":
            split = "validation"
        return load_dataset("xlangai/spider", split=split, streaming=args.dataset_streaming)
    
    
@datasources("xlangai/spider:stats")
class SpiderInstructWithStats(SpiderInstruct):
    SCHEMA_WITH_STATS = True


@datasources("chinmayc3/bird-sql")
class BirdSQL(BaseSQLDataSource):
    def get_schema_dataset(self, split: str):
        if split == "test":
            split = "dev"
        return load_dataset("iknow-lab/bird-schema", split=split)

    def get_sql_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        if split == "test":
            ds = load_dataset("heegyu/bird-sql-mini-dev", split="validation", streaming=args.dataset_streaming)
        else:
            ds = load_dataset("chinmayc3/bird-sql", split=split, streaming=args.dataset_streaming)
        ds = ds.rename_column("SQL", "query")
        return ds
        
@datasources("chinmayc3/bird-sql:stats")
class BirdSQLWithStats(BirdSQL):
    SCHEMA_WITH_STATS = True