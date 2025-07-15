async def forward_180(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_instr_0_1 = "Sub-task 0_1: Extract and summarize the key physical features of the solar neutrino production relevant to the problem, including the characteristics of the pp-III branch and its neutrino energy spectrum, and the definition of the two energy bands. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_0_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_0_1 = []
    all_answer_0_1 = []
    subtask_desc_0_1 = {
        "subtask_id": "subtask_0_1",
        "instruction": debate_instr_0_1,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(1):
        for i, agent in enumerate(debate_agents_0_1):
            thinking, answer = await agent([taskInfo], debate_instr_0_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, subtask_0_1, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_0_1.append(thinking)
            all_answer_0_1.append(answer)
    final_decision_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_1, answer_0_1 = await final_decision_agent_0_1([taskInfo] + all_thinking_0_1 + all_answer_0_1, "Sub-task 0_1: Extract and summarize key physical features and neutrino energy bands. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 0_1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0_1: ", sub_tasks[-1])

    cot_instruction_0_2 = "Sub-task 0_2: Clarify the temporal aspect of neutrino detection relative to the stoppage of the pp-III branch, confirming that neutrinos detected now correspond to emissions before the stoppage and implications for flux measurement."
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "subtask_0_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, subtask_0_2, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 0_2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0_2: ", sub_tasks[-1])

    cot_sc_instruction_1_1 = "Sub-task 1_1: Compute or identify the expected neutrino flux contributions from each proton-proton chain branch (pp-I, pp-II, pp-III) within the two specified energy bands (700-800 keV and 800-900 keV) under normal conditions, based on the output from Sub-task 0_1."
    N_sc_1_1 = self.max_sc
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_1)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "subtask_1_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_1, answer_0_1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_1):
        thinking_i, answer_i = await cot_agents_1_1[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, subtask_1_1, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_1.append(answer_i)
        possible_thinkings_1_1.append(thinking_i)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo, thinking_0_1, answer_0_1] + possible_thinkings_1_1 + possible_answers_1_1, "Sub-task 1_1: Synthesize and choose the most consistent neutrino flux contributions for each branch and energy band.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1_1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1_1: ", sub_tasks[-1])

    debate_instr_1_2 = "Sub-task 1_2: Determine how stopping the pp-III branch affects the neutrino flux in each energy band, considering that only pp-III neutrinos are removed while others remain unchanged, based on outputs from Sub-task 1_1 and Sub-task 0_2. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_1_2 = []
    all_answer_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "subtask_1_2",
        "instruction": debate_instr_1_2,
        "context": ["user query", thinking_1_1, answer_1_1, thinking_0_2, answer_0_2],
        "agent_collaboration": "Debate"
    }
    for r in range(1):
        for i, agent in enumerate(debate_agents_1_2):
            thinking, answer = await agent([taskInfo, thinking_1_1, answer_1_1, thinking_0_2, answer_0_2], debate_instr_1_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, subtask_1_2, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_1_2.append(thinking)
            all_answer_1_2.append(answer)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo, thinking_1_1, answer_1_1, thinking_0_2, answer_0_2] + all_thinking_1_2 + all_answer_1_2, "Sub-task 1_2: Synthesize and choose the most consistent answer on how stopping pp-III affects neutrino flux in each band.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1_2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1_2: ", sub_tasks[-1])

    cot_sc_instruction_2_1 = "Sub-task 2_1: Combine the flux contributions from the remaining branches to calculate the adjusted fluxes in band 1 and band 2 after the pp-III branch stoppage, and compute the ratio Flux(band 1) / Flux(band 2), based on output from Sub-task 1_2."
    N_sc_2_1 = self.max_sc
    cot_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_2_1)]
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    subtask_desc_2_1 = {
        "subtask_id": "subtask_2_1",
        "instruction": cot_sc_instruction_2_1,
        "context": ["user query", thinking_1_2, answer_1_2],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_2_1):
        thinking_i, answer_i = await cot_agents_2_1[i]([taskInfo, thinking_1_2, answer_1_2], cot_sc_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_1[i].id}, subtask_2_1, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_2_1.append(answer_i)
        possible_thinkings_2_1.append(thinking_i)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo, thinking_1_2, answer_1_2] + possible_thinkings_2_1 + possible_answers_2_1, "Sub-task 2_1: Synthesize and choose the most consistent adjusted flux ratio after pp-III stoppage.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2_1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2_1: ", sub_tasks[-1])

    debate_instr_3_1 = "Sub-task 3_1: Select the approximate ratio of fluxes from the given choices (0.1, 10, 1, 0.01) based on the computed ratio from Sub-task 2_1 and justify the selection. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking_3_1 = []
    all_answer_3_1 = []
    subtask_desc_3_1 = {
        "subtask_id": "subtask_3_1",
        "instruction": debate_instr_3_1,
        "context": ["user query", thinking_2_1, answer_2_1],
        "agent_collaboration": "Debate"
    }
    for r in range(1):
        for i, agent in enumerate(debate_agents_3_1):
            thinking, answer = await agent([taskInfo, thinking_2_1, answer_2_1], debate_instr_3_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, subtask_3_1, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_3_1.append(thinking)
            all_answer_3_1.append(answer)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1([taskInfo, thinking_2_1, answer_2_1] + all_thinking_3_1 + all_answer_3_1, "Sub-task 3_1: Synthesize and choose the best approximate ratio and justify it.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3_1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 3_1: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_1, answer_3_1, sub_tasks, agents)
    return final_answer, logs
