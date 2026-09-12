import React, { useEffect, useState } from 'react';
import { Select } from 'antd';

interface AgentSelectorProps {
  value: string;
  onChange: (value: string) => void;
  // 列表加载后的自动补默认值走这个 —— 它不该触发「选 agent 即新开对话」
  // 的用户路径（否则直接打开会话链接时历史会被清空、URL 被劫持到 /chat）。
  onBootstrap: (value: string) => void;
}

interface AgentOption {
  value: string;
  label: string;
}

// 可选的 agent 从后端注册表动态取（GET /agents，见 ai/agent/agents.py 的
// get_all_agent_info）。之前这里是注册表的**手工镜像** —— 两边必须同时改，
// 否则会出现「下拉里能选、后端不认识」或反过来；agent 改名时这个缺口真咬过人。
//
// 拿不到列表时不兜底造假数据：下拉为空 + placeholder 说明原因。
// 此时用户仍能发消息 —— 请求里省略 agent_id，后端用 DEFAULT_AGENT（见
// useStreamChat.ts）。宁可空着，也不要一份会漂移的副本。
const AgentSelector: React.FC<AgentSelectorProps> = ({ value, onChange, onBootstrap }) => {
  const [options, setOptions] = useState<AgentOption[]>([]);

  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_BASE_URL}/agents`)
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      })
      .then((list) =>
        setOptions(
          list.map((agent: { key: string; description: string }) => {
            const [name, desc = ""] = agent.description.split("：");
            return {
              value: agent.key,
              title: name,
              label: (
                <div style={{ lineHeight: 1.4, padding: "2px 0" }}>
                  <div style={{ fontWeight: 500 }}>{name}</div>
                  {desc && <div style={{ fontSize: 12, color: "#6e6e73" }}>{desc}</div>}
                </div>
              ),
            };
          })
        )
      )
      .catch((err) => console.error("加载 agent 列表失败", err));
  }, []);

  // 列表到达时，如果当前值不在列表里（包括初始的空值），自动选第一个。
  // 这样「默认用哪个 agent」也由后端注册表的顺序决定，前端不再硬编码。
  useEffect(() => {
    if (options.length > 0 && !options.some((option) => option.value === value)) {
      onBootstrap(options[0].value);
    }
  }, [options, value, onChange]);

  return (
    <Select
      value={value || undefined}
      className="ml-2 mr-5 w-44"
      onChange={onChange}
      options={options}
      optionLabelProp="title"   // 选中态只显示短名（如「快速问答」），全说明留在下拉里
      placeholder={options.length === 0 ? "列表加载中…" : undefined}
      notFoundContent="后端 agent 列表不可用"
    />
  );
};

export default AgentSelector;
