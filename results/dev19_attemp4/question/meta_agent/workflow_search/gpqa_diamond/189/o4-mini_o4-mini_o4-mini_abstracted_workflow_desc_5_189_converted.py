async def forward_189(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: SC_CoT - Extract and summarize relevant information
    cot_sc_instruction0 = (
        "Sub-task 0: Extract and summarize all relevant information from the query, "
        "including the list of nucleophiles, solvent context (aqueous solution), "
        "and underlying factors that affect nucleophilicity (basicity, polarizability, solvation, resonance)."
    )
    N0 = self.max_sc
    cot_agents0 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N0)]
    possible_thinkings0 = []
    possible_answers0 = []
    subtask_desc0 = {"subtask_id": "subtask_0", "instruction": cot_sc_instruction0, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for i in range(N0):
        thinking_i, answer_i = await cot_agents0[i]([taskInfo], cot_sc_instruction0, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents0[i].id}, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_thinkings0.append(thinking_i)
        possible_answers0.append(answer_i)
    final_decision_agent0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr0 = "Given all the above thinking and answers, find the most consistent and correct summary for the extracted information."
    thinking0, answer0 = await final_decision_agent0(
        [taskInfo] + possible_thinkings0 + possible_answers0,
        "Sub-task 0: Synthesize and choose the most consistent summary. " + final_instr0,
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 0 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc0)
    print("Step 1: ", sub_tasks[-1])

    # Stage 1: SC_CoT - Analyze and classify each nucleophile
    cot_sc_instruction1 = (
        "Sub-task 1: Analyze and classify each nucleophile (4-methylcyclohexan-1-olate, hydroxide, propionate, methanol, ethanethiolate) "
        "according to their key attributes: charge, basicity, polarizability, resonance delocalization, and degree of solvation in water."
    )
    N1 = self.max_sc
    cot_agents1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_sc_instruction1, "context": ["user query", "thinking of subtask 0", "answer of subtask 0"], "agent_collaboration": "SC_CoT"}
    for i in range(N1):
        thinking_i, answer_i = await cot_agents1[i]([taskInfo, thinking0, answer0], cot_sc_instruction1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents1[i].id}, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_thinkings1.append(thinking_i)
        possible_answers1.append(answer_i)
    final_decision_agent1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr1 = "Given all the above thinking and answers, find the most consistent and correct classification."
    thinking1, answer1 = await final_decision_agent1(
        [taskInfo, thinking0, answer0] + possible_thinkings1 + possible_answers1,
        "Sub-task 1: Synthesize and choose the most consistent classification. " + final_instr1,
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 2: ", sub_tasks[-1])

    # Stage 2: Debate - Generate candidate rankings
    instruction2_base = (
        "Sub-task 2: Define transformation criteria (e.g., ordering by decreasing basicity, increasing polarizability, "
        "decreasing solvation, and resonance effects) and generate candidate ranking variants of the five nucleophiles based on these factors."
    )
    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction2 = instruction2_base + " " + debate_instr
    debate_agents2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking2 = [[] for _ in range(self.max_round)]
    all_answer2 = [[] for _ in range(self.max_round)]
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": debate_instruction2, "context": ["user query", "thinking of subtask 1", "answer of subtask 1"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents2):
            if r == 0:
                thinking_r, answer_r = await agent([taskInfo, thinking1, answer1], debate_instruction2, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking1, answer1] + all_thinking2[r-1] + all_answer2[r-1]
                thinking_r, answer_r = await agent(input_infos, debate_instruction2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking_r.content}; answer: {answer_r.content}")
            all_thinking2[r].append(thinking_r)
            all_answer2[r].append(answer_r)
    final_decision_agent2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr2 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking2, answer2 = await final_decision_agent2(
        [taskInfo, thinking1, answer1] + all_thinking2[-1] + all_answer2[-1],
        "Sub-task 2: " + instruction2_base + " " + final_instr2,
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 3: ", sub_tasks[-1])

    # Stage 3: Debate - Evaluate and select final order
    instruction3_base = (
        "Sub-task 3: Evaluate the candidate rankings against empirical nucleophilicity trends in water and the provided answer choices "
        "to select the most consistent order from most to least reactive."
    )
    debate_instruction3 = instruction3_base + " " + debate_instr
    debate_agents3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking3 = [[] for _ in range(self.max_round)]
    all_answer3 = [[] for _ in range(self.max_round)]
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": debate_instruction3, "context": ["user query", "thinking of subtask 2", "answer of subtask 2"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents3):
            if r == 0:
                thinking_r, answer_r = await agent([taskInfo, thinking2, answer2], debate_instruction3, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
                thinking_r, answer_r = await agent(input_infos, debate_instruction3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking_r.content}; answer: {answer_r.content}")
            all_thinking3[r].append(thinking_r)
            all_answer3[r].append(answer_r)
    final_decision_agent3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr3 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking3, answer3 = await final_decision_agent3(
        [taskInfo, thinking2, answer2] + all_thinking3[-1] + all_answer3[-1],
        "Sub-task 3: " + instruction3_base + " " + final_instr3,
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs