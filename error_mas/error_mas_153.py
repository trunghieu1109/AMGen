async def forward_153(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0_1 = "Sub-task 1: Extract and summarize the given spectral information to understand the key features of the unidentified compound."
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                                model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0_subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, analyzing spectral data, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {
        "thinking": thinking_0_1,
        "answer": answer_0_1
    }
    logs.append(subtask_desc_0_1)
    print("Step 1: ", sub_tasks[-1])

    debate_instruction_0_2 = "Sub-task 2: Generate possible structural variants based on the extracted spectral data, considering the presence of chlorine, O-H, and carbonyl groups."
    debate_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                     model=self.node_model, role=role, temperature=0.5) 
                        for role in self.debate_role]
    N_max_0_2 = self.max_round
    all_thinking_0_2 = [[] for _ in range(N_max_0_2)]
    all_answer_0_2 = [[] for _ in range(N_max_0_2)]
    subtask_desc_0_2 = {
        "subtask_id": "stage_0_subtask_2",
        "instruction": debate_instruction_0_2,
        "context": ["user query", "thinking of stage_0_subtask_1", "answer of stage_0_subtask_1"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_0_2):
        for i, agent in enumerate(debate_agents_0_2):
            if r == 0:
                thinking_0_2, answer_0_2 = await agent([taskInfo, thinking_0_1, answer_0_1], 
                                               debate_instruction_0_2, r, is_sub_task=True)
            else:
                input_infos_0_2 = [taskInfo, thinking_0_1, answer_0_1] + all_thinking_0_2[r-1] + all_answer_0_2[r-1]
                thinking_0_2, answer_0_2 = await agent(input_infos_0_2, debate_instruction_0_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, generating structural variants, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
            all_thinking_0_2[r].append(thinking_0_2)
            all_answer_0_2[r].append(answer_0_2)
    final_instr_0_2 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                           model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo, thinking_0_1, answer_0_1] + all_thinking_0_2[-1] + all_answer_0_2[-1], 
                                                     "Sub-task 2: Generate structural variants" + final_instr_0_2, 
                                                     is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {
        "thinking": thinking_0_2,
        "answer": answer_0_2
    }
    logs.append(subtask_desc_0_2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_1_1 = "Sub-task 1: Assess the impact of the presence of chlorine and other functional groups on the molecular structure and weight."
    N_1_1 = self.max_sc
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                                  model=self.node_model, temperature=0.5) for _ in range(N_1_1)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1_subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", "thinking of stage_0_subtask_2", "answer of stage_0_subtask_2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_1_1):
        thinking_1_1, answer_1_1 = await cot_agents_1_1([taskInfo, thinking_0_2, answer_0_2], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, assessing impact of functional groups, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
        possible_answers_1_1.append(answer_1_1)
        possible_thinkings_1_1.append(thinking_1_1)
    final_instr_1_1 = "Given all the above thinking and answers, find the most consistent and correct solutions for the impact assessment."
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                           model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo, thinking_0_2, answer_0_2] + possible_thinkings_1_1 + possible_answers_1_1, 
                                                     "Sub-task 1: Assess impact of functional groups" + final_instr_1_1, 
                                                     is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {
        "thinking": thinking_1_1,
        "answer": answer_1_1
    }
    logs.append(subtask_desc_1_1)
    print("Step 3: ", sub_tasks[-1])

    debate_instruction_2_1 = "Sub-task 1: Evaluate the generated structural variants against the given choices to identify the most likely structure."
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                     model=self.node_model, role=role, temperature=0.5) 
                        for role in self.debate_role]
    N_max_2_1 = self.max_round
    all_thinking_2_1 = [[] for _ in range(N_max_2_1)]
    all_answer_2_1 = [[] for _ in range(N_max_2_1)]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2_subtask_1",
        "instruction": debate_instruction_2_1,
        "context": ["user query", "thinking of stage_1_subtask_1", "answer of stage_1_subtask_1"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_1):
        for i, agent in enumerate(debate_agents_2_1):
            if r == 0:
                thinking_2_1, answer_2_1 = await agent([taskInfo, thinking_1_1, answer_1_1], 
                                               debate_instruction_2_1, r, is_sub_task=True)
            else:
                input_infos_2_1 = [taskInfo, thinking_1_1, answer_1_1] + all_thinking_2_1[r-1] + all_answer_2_1[r-1]
                thinking_2_1, answer_2_1 = await agent(input_infos_2_1, debate_instruction_2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating structural variants, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
            all_thinking_2_1[r].append(thinking_2_1)
            all_answer_2_1[r].append(answer_2_1)
    final_instr_2_1 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                           model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo, thinking_1_1, answer_1_1] + all_thinking_2_1[-1] + all_answer_2_1[-1], 
                                                     "Sub-task 1: Evaluate structural variants" + final_instr_2_1, 
                                                     is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {
        "thinking": thinking_2_1,
        "answer": answer_2_1
    }
    logs.append(subtask_desc_2_1)
    print("Step 4: ", sub_tasks[-1])

    cot_sc_instruction_2_2 = "Sub-task 2: Select and prioritize the structural variant that best conforms to the spectral data and matches one of the given choices."
    N_2_2 = self.max_sc
    cot_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                                  model=self.node_model, temperature=0.5) for _ in range(N_2_2)]
    possible_answers_2_2 = []
    possible_thinkings_2_2 = []
    subtask_desc_2_2 = {
        "subtask_id": "stage_2_subtask_2",
        "instruction": cot_sc_instruction_2_2,
        "context": ["user query", "thinking of stage_2_subtask_1", "answer of stage_2_subtask_1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_2_2):
        thinking_2_2, answer_2_2 = await cot_agents_2_2([taskInfo, thinking_2_1, answer_2_1], cot_sc_instruction_2_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_2[i].id}, selecting best structural variant, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
        possible_answers_2_2.append(answer_2_2)
        possible_thinkings_2_2.append(thinking_2_2)
    final_instr_2_2 = "Given all the above thinking and answers, find the most consistent and correct solutions for the selection."
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                           model=self.node_model, temperature=0.0)
    thinking_2_2, answer_2_2 = await final_decision_agent_2_2([taskInfo, thinking_2_1, answer_2_1] + possible_thinkings_2_2 + possible_answers_2_2, 
                                                     "Sub-task 2: Select best structural variant" + final_instr_2_2, 
                                                     is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {
        "thinking": thinking_2_2,
        "answer": answer_2_2
    }
    logs.append(subtask_desc_2_2)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_2, answer_2_2, sub_tasks, agents)
    return final_answer, logs
