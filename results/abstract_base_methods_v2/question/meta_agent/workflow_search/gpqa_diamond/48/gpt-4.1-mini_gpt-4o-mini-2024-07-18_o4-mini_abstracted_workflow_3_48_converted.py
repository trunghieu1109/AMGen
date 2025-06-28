async def forward_48(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = "Sub-task 1: Analyze the general principles and mechanistic features of sigmatropic rearrangements, including the migration of pi bonds into sigma bonds, and summarize their thermodynamic favorability and typical reaction conditions."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing sigmatropic rearrangements, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Subtask 1 answer: ", sub_tasks[-1])
    debate_instruction_2 = "Sub-task 2: Review and distinguish the specific characteristics, mechanisms, and typical outcomes of Cope and Claisen rearrangements, emphasizing their structural changes and relevance to the given substrates."
    debate_agents_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2 = self.max_round
    all_thinking2 = [[] for _ in range(N_max_2)]
    all_answer2 = [[] for _ in range(N_max_2)]
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": debate_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2):
        for i, agent in enumerate(debate_agents_2):
            input_infos_2 = [taskInfo, thinking1, answer1]
            if r > 0:
                input_infos_2.extend(all_thinking2[r-1])
            thinking2, answer2 = await agent(input_infos_2, debate_instruction_2, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, reviewing Cope and Claisen rearrangements, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking2[r].append(thinking2)
            all_answer2[r].append(answer2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + all_thinking2[-1] + all_answer2[-1], "Sub-task 2: Make a final decision on Cope and Claisen rearrangements characteristics.", is_sub_task=True)
    agents.append(f"Final Decision agent on Cope and Claisen rearrangements, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Subtask 2 answer: ", sub_tasks[-1])
    N = self.max_sc
    subtask3_products = []
    subtask3_thinking_map = {}
    subtask3_answer_map = {}
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": "Sub-task 3: For the first reaction (1,1-dimethoxyethan-1-amine + but-3-en-2-ol + H+ + Heat), generate and evaluate multiple mechanistic hypotheses including acid-catalyzed cyclization (e.g., aza-Prins), sigmatropic rearrangement, and ionic pathways; determine the most plausible product A based on mechanistic reasoning and literature precedents.",
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2, answer2], subtask_desc3["instruction"], is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, evaluating mechanistic hypotheses for product A, thinking: {thinking3.content}; answer: {answer3.content}")
        subtask3_products.append(answer3.content)
        subtask3_thinking_map[answer3.content] = thinking3
        subtask3_answer_map[answer3.content] = answer3
    most_common_answer_3 = Counter(subtask3_products).most_common(1)[0][0]
    thinking3 = subtask3_thinking_map[most_common_answer_3]
    answer3 = subtask3_answer_map[most_common_answer_3]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Subtask 3 answer: ", sub_tasks[-1])
    subtask4_products = []
    subtask4_thinking_map = {}
    subtask4_answer_map = {}
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": "Sub-task 4: For the second reaction ((3R,4S)-3,4-dimethylhexa-1,5-diyne + Heat), independently analyze possible thermal reaction pathways including sigmatropic rearrangements, cyclizations, and alternative thermal rearrangements; generate multiple mechanistic hypotheses and select the most chemically plausible product B.",
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    for i in range(N):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking2, answer2], subtask_desc4["instruction"], is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, evaluating mechanistic hypotheses for product B, thinking: {thinking4.content}; answer: {answer4.content}")
        subtask4_products.append(answer4.content)
        subtask4_thinking_map[answer4.content] = thinking4
        subtask4_answer_map[answer4.content] = answer4
    most_common_answer_4 = Counter(subtask4_products).most_common(1)[0][0]
    thinking4 = subtask4_thinking_map[most_common_answer_4]
    answer4 = subtask4_answer_map[most_common_answer_4]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Subtask 4 answer: ", sub_tasks[-1])
    subtask5_products = []
    subtask5_thinking_map = {}
    subtask5_answer_map = {}
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": "Sub-task 5: For the third reaction (2-((vinyloxy)methyl)but-1-ene + Heat), explore mechanistic possibilities including vinyl ether Claisen rearrangement, acid-catalyzed pathways, and ionic mechanisms; generate multiple hypotheses and determine the most likely product C, carefully distinguishing between aldehyde and allylic alcohol outcomes.",
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    cot_agents_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    for i in range(N):
        thinking5, answer5 = await cot_agents_5[i]([taskInfo, thinking2, answer2], subtask_desc5["instruction"], is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5[i].id}, evaluating mechanistic hypotheses for product C, thinking: {thinking5.content}; answer: {answer5.content}")
        subtask5_products.append(answer5.content)
        subtask5_thinking_map[answer5.content] = thinking5
        subtask5_answer_map[answer5.content] = answer5
    most_common_answer_5 = Counter(subtask5_products).most_common(1)[0][0]
    thinking5 = subtask5_thinking_map[most_common_answer_5]
    answer5 = subtask5_answer_map[most_common_answer_5]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Subtask 5 answer: ", sub_tasks[-1])
    debate_instruction_6 = "Sub-task 6: Conduct a reflexive cross-examination and debate of the proposed products A, B, and C from subtasks 3, 4, and 5, comparing them against known chemical precedents, mechanistic plausibility, and the multiple-choice options; iteratively refine product assignments to ensure chemical correctness and consistency."
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": debate_instruction_6,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            input_infos_6 = [taskInfo, thinking3, answer3, thinking4, answer4, thinking5, answer5]
            if r > 0:
                input_infos_6.extend(all_thinking6[r-1])
            thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating product assignments, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make a final decision on the refined and validated products A, B, and C.", is_sub_task=True)
    agents.append(f"Final Decision agent on product validation, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)
    print("Subtask 6 answer: ", sub_tasks[-1])
    subtask7_instruction = "Sub-task 7: Select the correct multiple-choice answer (A, B, C, or D) that corresponds to the refined and validated products A, B, and C determined in subtask 6, ensuring strict adherence to output formatting requirements."
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": subtask7_instruction,
        "context": ["user query", "thinking of subtask 6", "answer of subtask 6"],
        "agent_collaboration": "CoT"
    }
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6, answer6], subtask7_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, selecting final multiple-choice answer, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {
        "thinking": thinking7,
        "answer": answer7
    }
    logs.append(subtask_desc7)
    print("Subtask 7 answer: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs