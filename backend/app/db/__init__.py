"""数据库包。

这里曾经在**导入时**就创建一个没有任何代码引用的 SQLite engine
（sqlite:///./sql_app.db）和一个 Base —— 属于 FastAPI 教程脚手架的残留。
两者都已移除。

真正的数据库连接、会话工厂与建表逻辑都在 db/database.py。
"""
