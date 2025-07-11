async def forward_156(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = "Sub-task 1: Extract the core question for designing a molecular diagnostic kit for quick detection of a retrovirus during an outbreak."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "stage_1.subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    cot_sc_instruction = "Sub-task 2: Parse and catalog each of the four answer choices, noting the detection method and sequencing approach they propose."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinking_mapping = {}
    answer_mapping = {}
    subtask_desc2 = {"subtask_id": "stage_1.subtask_2", "instruction": cot_sc_instruction, "context": ["user query", sub_tasks[-1]], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinking_mapping[answer2.content] = thinking2
        answer_mapping[answer2.content] = answer2
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinking_mapping[answer2_content]
    answer2 = answer_mapping[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    cot_reflect_instruction = "Sub-task 3: Identify the key constraints and requirements: retroviral nature (RNA genome), speed, molecular specificity, and accuracy."
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    subtask_desc3 = {"subtask_id": "stage_1.subtask_3", "instruction": cot_reflect_instruction, "context": ["user query", sub_tasks[-2], sub_tasks[-1]], "agent_collaboration": "Reflexion"}
    thinking3, answer3 = await cot_agent3([taskInfo, thinking1, answer1, thinking2, answer2], cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3], "Please review the identified constraints and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        thinking3, answer3 = await cot_agent3([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, feedback], cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    cot_instruction4 = "Sub-task 4: Define evaluation criteria based on constraints: suitability for RNA viruses (reverse transcription), assay speed, sensitivity, and workflow feasibility."
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {"subtask_id": "stage_2.subtask_1", "instruction": cot_instruction4, "context": ["user query", sub_tasks[-1]], "agent_collaboration": "CoT"}
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    cot_sc_instruction5 = "Sub-task 5: Evaluate each parsed choice against the criteria: check RNA handling via cDNA, compare PCR vs ELISA speed and specificity, and sequencing relevance."
    cot_agents5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers5 = []
    thinking_mapping5 = {}
    answer_mapping5 = {}
    subtask_desc5 = {"subtask_id": "stage_2.subtask_2", "instruction": cot_sc_instruction5, "context": ["user query", sub_tasks[-4], sub_tasks[-3], sub_tasks[-2], sub_tasks[-1]], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking5, answer5 = await cot_agents5[i]([taskInfo, thinking2, answer2, thinking4, answer4], cot_sc_instruction5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents5[i].id}, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers5.append(answer5.content)
        thinking_mapping5[answer5.content] = thinking5
        answer_mapping5[answer5.content] = answer5
    answer5_content = Counter(possible_answers5).most_common(1)[0][0]
    thinking5 = thinking_mapping5[answer5_content]
    answer5 = answer_mapping5[answer5_content]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    debate_instruction = "Sub-task 6: Rank the choices by scoring them on each criterion to determine which method best meets the quick, accurate, retroviral detection requirements."
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking = [[] for _ in range(N_max)]
    all_answer = [[] for _ in range(N_max)]
    subtask_desc6 = {"subtask_id": "stage_2.subtask_3", "instruction": debate_instruction, "context": ["user query", sub_tasks[-3], sub_tasks[-2], sub_tasks[-1]], "agent_collaboration": "Debate"}
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking2, answer2, thinking4, answer4, thinking5, answer5], debate_instruction, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking2, answer2, thinking4, answer4, thinking5, answer5] + all_thinking[r-1] + all_answer[r-1]
                thinking6, answer6 = await agent(inputs, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking[r].append(thinking6)
            all_answer[r].append(answer6)
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent([taskInfo] + all_thinking[-1] + all_answer[-1], "Sub-task 6: Make final decision on the ranking.", is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    cot_instruction7 = "Sub-task 7: Select the highest-ranked choice and format the output as the single-letter answer."
    cot_agent7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {"subtask_id": "stage_2.subtask_4", "instruction": cot_instruction7, "context": ["user query", sub_tasks[-1]], "agent_collaboration": "CoT"}
    thinking7, answer7 = await cot_agent7([taskInfo, thinking6, answer6], cot_instruction7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent7.id}, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs