import React from 'react';
import { Select } from 'antd';

interface AgentSelectorProps {
  value: string;
  onChange: (value: string) => void;
}

// 可选的 agent。
//
// ⚠️ 这是后端 ai/agent/agents.py 里那张注册表的**手工镜像** —— 两边必须同时改，
// 否则会出现「下拉里能选、后端不认识」或者反过来「后端有、选不到」。
// 改造 #10 会把这个列表改成从后端 get_all_agent_info() 动态取。
//
// research-workflow 是改造 #3 的产物（Corrective RAG：分析 → 检索 → 判证据充分性
// → 补充检索 → 综合），和 oa-assistant（ReAct 自由循环）共用同一套工具与检索管线，
// 并存就是为了能直接对比。
const AGENTS = [
  { value: "research-workflow", label: "RESEARCH-WORKFLOW" },
  { value: "oa-assistant", label: "OA-ASSISTANT" },
  { value: "multi-agent-supervisor", label: "MULTI-AGENT-SUPERVISOR" },
];

const AgentSelector: React.FC<AgentSelectorProps> = ({ value, onChange }) => {
  return (
    <Select
      value={value}
      className="ml-2 mr-5 w-44"
      onChange={onChange}
      options={AGENTS}
    />
  );
};

export default AgentSelector;
