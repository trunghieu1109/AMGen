async def forward_156(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = "Sub-task 1: Confirm the outbreak involves a retrovirus by analyzing epidemiological data, clinical symptoms, and literature on similar viral outbreaks."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, confirming retrovirus presence, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    cot_instruction2 = "Sub-task 2: Collect patient blood or swab samples under biosafety conditions and extract high-quality total RNA."
    cot_agent2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_instruction2, "context": ["user query", "thinking1.content", "answer1.content"], "agent_collaboration": "CoT"}
    thinking2, answer2 = await cot_agent2([taskInfo, thinking1, answer1], cot_instruction2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent2.id}, collecting samples and extracting RNA, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    cot_instruction3 = "Sub-task 3: Perform reverse transcription to generate complementary DNA (cDNA) from extracted RNA."
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_instruction3, "context": ["user query", "thinking2.content", "answer2.content"], "agent_collaboration": "CoT"}
    thinking3, answer3 = await cot_agent3([taskInfo, thinking2, answer2], cot_instruction3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3.id}, performing reverse transcription, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    cot_sc_instruction4 = "Sub-task 4: Design primers and fluorescent probes targeting a conserved region of the retroviral genome with in silico specificity checks."
    N = self.max_sc
    cot_agents4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers4 = []
    thinking_mapping4 = {}
    answer_mapping4 = {}
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_sc_instruction4, "context": ["user query", "thinking3.content", "answer3.content"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking4_i, answer4_i = await cot_agents4[i]([taskInfo, thinking3, answer3], cot_sc_instruction4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents4[i].id}, designing primers and probes, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
        possible_answers4.append(answer4_i.content)
        thinking_mapping4[answer4_i.content] = thinking4_i
        answer_mapping4[answer4_i.content] = answer4_i
    answer4_content = Counter(possible_answers4).most_common(1)[0][0]
    thinking4 = thinking_mapping4[answer4_content]
    answer4 = answer_mapping4[answer4_content]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    debate_instruction5 = "Sub-task 5: Develop and optimize a real-time PCR protocol by evaluating primer/probe concentrations, annealing temperatures, and cycling conditions."
    debate_agents5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking5 = [[] for _ in range(self.max_round)]
    all_answer5 = [[] for _ in range(self.max_round)]
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": debate_instruction5, "context": ["user query", "thinking4.content", "answer4.content"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents5):
            if r == 0:
                thinking5_i, answer5_i = await agent([taskInfo, thinking4, answer4], debate_instruction5, r, is_sub_task=True)
            else:
                thinking5_i, answer5_i = await agent([taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1], debate_instruction5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, optimizing PCR protocol, thinking: {thinking5_i.content}; answer: {answer5_i.content}")
            all_thinking5[r].append(thinking5_i)
            all_answer5[r].append(answer5_i)
    final_decision_agent5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Select the optimal PCR protocol configuration.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting PCR protocol, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    cot_reflect_instruction6 = "Sub-task 6: Validate the real-time PCR assay using positive and negative samples, determining sensitivity, specificity, and limit of detection."
    cot_agent6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max6 = self.max_round
    subtask_desc6 = {"subtask_id": "subtask_6", "instruction": cot_reflect_instruction6, "context": ["user query", "thinking5.content", "answer5.content"], "agent_collaboration": "Reflexion"}
    thinking6, answer6 = await cot_agent6([taskInfo, thinking5, answer5], cot_reflect_instruction6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent6.id}, initial validation design, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max6):
        feedback6, correct6 = await critic_agent6([taskInfo, thinking6, answer6], "Please review the validation design and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent6.id}, providing feedback, thinking: {feedback6.content}; answer: {correct6.content}")
        if correct6.content == "True":
            break
        thinking6, answer6 = await cot_agent6([taskInfo, thinking5, answer5, thinking6, answer6, feedback6], cot_reflect_instruction6, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent6.id}, refining validation design, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    cot_sc_instruction7 = "Sub-task 7: Formulate the diagnostic kit by lyophilizing mastermix reagents and preparing stable reaction buffers."
    N7 = self.max_sc
    cot_agents7 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N7)]
    possible_answers7 = []
    thinking_mapping7 = {}
    answer_mapping7 = {}
    subtask_desc7 = {"subtask_id": "subtask_7", "instruction": cot_sc_instruction7, "context": ["user query", "thinking6.content", "answer6.content"], "agent_collaboration": "SC_CoT"}
    for i in range(N7):
        thinking7_i, answer7_i = await cot_agents7[i]([taskInfo, thinking6, answer6], cot_sc_instruction7, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents7[i].id}, formulating kit reagents, thinking: {thinking7_i.content}; answer: {answer7_i.content}")
        possible_answers7.append(answer7_i.content)
        thinking_mapping7[answer7_i.content] = thinking7_i
        answer_mapping7[answer7_i.content] = answer7_i
    answer7_content = Counter(possible_answers7).most_common(1)[0][0]
    thinking7 = thinking_mapping7[answer7_content]
    answer7 = answer_mapping7[answer7_content]
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    debate_instruction8 = "Sub-task 8: Create comprehensive kit documentation including SOPs, QC guidelines, troubleshooting tips, and regulatory compliance information."
    debate_agents8 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking8 = [[] for _ in range(self.max_round)]
    all_answer8 = [[] for _ in range(self.max_round)]
    subtask_desc8 = {"subtask_id": "subtask_8", "instruction": debate_instruction8, "context": ["user query", "thinking7.content", "answer7.content"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents8):
            if r == 0:
                thinking8_i, answer8_i = await agent([taskInfo, thinking7, answer7], debate_instruction8, r, is_sub_task=True)
            else:
                thinking8_i, answer8_i = await agent([taskInfo, thinking7, answer7] + all_thinking8[r-1] + all_answer8[r-1], debate_instruction8, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, drafting documentation, thinking: {thinking8_i.content}; answer: {answer8_i.content}")
            all_thinking8[r].append(thinking8_i)
            all_answer8[r].append(answer8_i)
    final_decision_agent8 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await final_decision_agent8([taskInfo] + all_thinking8[-1] + all_answer8[-1], "Sub-task 8: Finalize kit documentation.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing documentation, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print("Step 8: ", sub_tasks[-1])
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer, logs