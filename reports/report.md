测试目标：验证 Global Client 在主备切换场景下的可用性和延迟表现。                     

基础设施：                                                                        
- 2 个 Milvus Standalone（by-dev1:19530, by-dev2:19531）+ 2 个 CDC 进程 + Pulsar
- 1 个 Mock 拓扑 REST Server（模拟 Global Endpoint）
- 启动命令：milvus_control -m -u start_milvus

测试过程：
1. 初始化 CDC 双向复制，创建 Collection
2. 启动 insert 线程（50ms 间隔）和 search 线程（100ms 间隔，3s 预热）
3. 按计划执行 4 次主备切换（2 快 + 2 慢），切换通过 CDC API + 拓扑翻转实现
4. 等待测试时长结束，收集指标，生成报告

核心断言：
- 切换窗口外 insert/search 失败率 = 0
- 切换后 30 秒内自动恢复
- 所有切换操作成功

运行命令：
conda run -n milvus2 python -m pytest tests/e2e_global_client/ -v -s \
--test-duration=720 --switchover-interval=120 --switchover-window=15 \
--report-dir=./reports --keep-metrics-server --timeout=0

监控：Prometheus(9090) + Grafana(3000) 可视化 insert/search
的吞吐、延迟和切换事件

- Insert：共 12,299 次，成功率 95.27%，切换窗口外失败率 0%                        
- Search：共 5,433 次，全部成功，零失败
- 切换：4 次切换均成功，每次耗时约 0.3 秒                                         
- 所有失败均发生在切换窗口内，错误类型为                             
STREAMING_CODE_REPLICATE_VIOLATION（写操作被发送到了 secondary
节点），属于预期行为
- Search 在切换期间不受影响，因为读操作在 primary 和 secondary 上都可以正常执行
- 延迟表现正常：insert p50=6.7ms，search p50=26ms

结论：Global Client
在主备切换场景下表现符合预期，切换窗口外零失败，读写操作均能自动恢复。