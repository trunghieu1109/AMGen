async def forward_152(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1a = "Sub-task 1a: Analyze Michael addition reaction (A) with reactants dimethyl malonate + methyl (E)-3-(p-tolyl)acrylate + (NaOEt, EtOH). Identify nucleophile, electrophile, and site of nucleophilic attack based on Michael addition principles."
    cot_agent_1a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_1a, answer_1a = await cot_agent_1a([taskInfo], cot_instruction_1a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1a.id}, analyzing reaction (A), thinking: {thinking_1a.content}; answer: {answer_1a.content}")
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking_1a.content}; answer - {answer_1a.content}")
    subtask_desc_1a['response'] = {"thinking": thinking_1a, "answer": answer_1a}
    logs.append(subtask_desc_1a)
    print("Step 1a: ", sub_tasks[-1])
    
    cot_instruction_1b = "Sub-task 1b: Analyze Michael addition reaction (B) with reactants 1-(cyclohex-1-en-1-yl)piperidine + (E)-but-2-enenitrile + (MeOH, H3O+). Identify nucleophile, electrophile, and site of nucleophilic attack based on Michael addition principles."
    cot_agent_1b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_instruction_1b,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_1b, answer_1b = await cot_agent_1b([taskInfo], cot_instruction_1b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1b.id}, analyzing reaction (B), thinking: {thinking_1b.content}; answer: {answer_1b.content}")
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking_1b.content}; answer - {answer_1b.content}")
    subtask_desc_1b['response'] = {"thinking": thinking_1b, "answer": answer_1b}
    logs.append(subtask_desc_1b)
    print("Step 1b: ", sub_tasks[-1])
    
    cot_instruction_1c = "Sub-task 1c: Analyze Michael addition reaction (C) with reactants C + but-3-en-2-one + (KOH, H2O). Deduce identity of C using product 2-(3-oxobutyl)cyclohexane-1,3-dione and reaction conditions. Identify nucleophile and electrophile."
    cot_agent_1c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1c = {
        "subtask_id": "subtask_1c",
        "instruction": cot_instruction_1c,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_1c, answer_1c = await cot_agent_1c([taskInfo], cot_instruction_1c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1c.id}, analyzing reaction (C), thinking: {thinking_1c.content}; answer: {answer_1c.content}")
    sub_tasks.append(f"Sub-task 1c output: thinking - {thinking_1c.content}; answer - {answer_1c.content}")
    subtask_desc_1c['response'] = {"thinking": thinking_1c, "answer": answer_1c}
    logs.append(subtask_desc_1c)
    print("Step 1c: ", sub_tasks[-1])
    
    cot_sc_instruction_2a = "Sub-task 2a: Determine major final product structure for reaction (A) by applying Michael addition mechanism: nucleophile attack at β-carbon of α,β-unsaturated carbonyl compound, followed by protonation or tautomerization as needed."
    N = self.max_sc
    cot_agents_2a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2a = []
    thinkingmapping_2a = {}
    answermapping_2a = {}
    subtask_desc_2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_sc_instruction_2a,
        "context": ["user query", "thinking of subtask_1a", "answer of subtask_1a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking_i, answer_i = await cot_agents_2a[i]([taskInfo, thinking_1a, answer_1a], cot_sc_instruction_2a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2a[i].id}, determining product (A), thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2a.append(answer_i.content)
        thinkingmapping_2a[answer_i.content] = thinking_i
        answermapping_2a[answer_i.content] = answer_i
    answer_2a_content = Counter(possible_answers_2a).most_common(1)[0][0]
    thinking_2a = thinkingmapping_2a[answer_2a_content]
    answer_2a = answermapping_2a[answer_2a_content]
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking_2a.content}; answer - {answer_2a.content}")
    subtask_desc_2a['response'] = {"thinking": thinking_2a, "answer": answer_2a}
    logs.append(subtask_desc_2a)
    print("Step 2a: ", sub_tasks[-1])
    
    cot_sc_instruction_2b = "Sub-task 2b: Determine major final product structure for reaction (B) by applying Michael addition mechanism: nucleophile attack at β-carbon of α,β-unsaturated nitrile, followed by protonation or tautomerization as needed."
    cot_agents_2b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2b = []
    thinkingmapping_2b = {}
    answermapping_2b = {}
    subtask_desc_2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_sc_instruction_2b,
        "context": ["user query", "thinking of subtask_1b", "answer of subtask_1b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking_i, answer_i = await cot_agents_2b[i]([taskInfo, thinking_1b, answer_1b], cot_sc_instruction_2b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2b[i].id}, determining product (B), thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2b.append(answer_i.content)
        thinkingmapping_2b[answer_i.content] = thinking_i
        answermapping_2b[answer_i.content] = answer_i
    answer_2b_content = Counter(possible_answers_2b).most_common(1)[0][0]
    thinking_2b = thinkingmapping_2b[answer_2b_content]
    answer_2b = answermapping_2b[answer_2b_content]
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking_2b.content}; answer - {answer_2b.content}")
    subtask_desc_2b['response'] = {"thinking": thinking_2b, "answer": answer_2b}
    logs.append(subtask_desc_2b)
    print("Step 2b: ", sub_tasks[-1])
    
    cot_sc_instruction_2c = "Sub-task 2c: Determine major final product structure for reaction (C) by applying Michael addition mechanism: nucleophile attack at β-carbon of but-3-en-2-one by enolate form of C, followed by protonation or tautomerization as needed."
    cot_agents_2c = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2c = []
    thinkingmapping_2c = {}
    answermapping_2c = {}
    subtask_desc_2c = {
        "subtask_id": "subtask_2c",
        "instruction": cot_sc_instruction_2c,
        "context": ["user query", "thinking of subtask_1c", "answer of subtask_1c"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking_i, answer_i = await cot_agents_2c[i]([taskInfo, thinking_1c, answer_1c], cot_sc_instruction_2c, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2c[i].id}, determining product (C), thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2c.append(answer_i.content)
        thinkingmapping_2c[answer_i.content] = thinking_i
        answermapping_2c[answer_i.content] = answer_i
    answer_2c_content = Counter(possible_answers_2c).most_common(1)[0][0]
    thinking_2c = thinkingmapping_2c[answer_2c_content]
    answer_2c = answermapping_2c[answer_2c_content]
    sub_tasks.append(f"Sub-task 2c output: thinking - {thinking_2c.content}; answer - {answer_2c.content}")
    subtask_desc_2c['response'] = {"thinking": thinking_2c, "answer": answer_2c}
    logs.append(subtask_desc_2c)
    print("Step 2c: ", sub_tasks[-1])
    
    debate_instruction_3 = "Sub-task 3: Compare deduced products for reactions (A), (B), and (C) with given multiple-choice options (choices 1 to 4). Identify which choice correctly matches all three products."
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking_3 = [[] for _ in range(N_max_3)]
    all_answer_3 = [[] for _ in range(N_max_3)]
    subtask_desc_3 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instruction_3,
        "context": ["user query", "thinking of subtask_2a", "answer of subtask_2a", "thinking of subtask_2b", "answer of subtask_2b", "thinking of subtask_2c", "answer of subtask_2c"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking_3, answer_3 = await agent([taskInfo, thinking_2a, answer_2a, thinking_2b, answer_2b, thinking_2c, answer_2c], debate_instruction_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking_2a, answer_2a, thinking_2b, answer_2b, thinking_2c, answer_2c] + all_thinking_3[r-1] + all_answer_3[r-1]
                thinking_3, answer_3 = await agent(input_infos_3, debate_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing products and matching choices, thinking: {thinking_3.content}; answer: {answer_3.content}")
            all_thinking_3[r].append(thinking_3)
            all_answer_3[r].append(answer_3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3, answer_3 = await final_decision_agent_3([taskInfo] + all_thinking_3[-1] + all_answer_3[-1], "Sub-task 3: Make final decision on correct choice matching all three products.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding correct choice, thinking: {thinking_3.content}; answer: {answer_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3['response'] = {"thinking": thinking_3, "answer": answer_3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking_3, answer_3, sub_tasks, agents)
    return final_answer, logs
