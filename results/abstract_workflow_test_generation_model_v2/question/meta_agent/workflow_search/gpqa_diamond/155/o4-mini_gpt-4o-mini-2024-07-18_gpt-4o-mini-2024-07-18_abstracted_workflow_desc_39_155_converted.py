async def forward_155(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = "Sub-task 1: Identify the structures with substituents on the double bond carbons of (E)-oct-4-ene and (Z)-oct-4-ene."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, identifying alkene structures, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction = "Sub-task 2: Explain the stereospecificity of mCPBA epoxidation and how it retains alkene configuration in the epoxide product."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmap = {}
    answermap = {}
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction, "context": ["user query", "subtask_1"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, explaining mCPBA stereospecificity, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_answers.append(answer2_i.content)
        thinkingmap[answer2_i.content] = thinking2_i
        answermap[answer2_i.content] = answer2_i
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmap[answer2_content]
    answer2 = answermap[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot_reflect_instruction = "Sub-task 3: Assign relative stereochemistry of epoxides: predict trans-epoxide from E-alkene and cis-epoxide from Z-alkene."
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_reflect_instruction, "context": ["user query", "subtask_1", "subtask_2"], "agent_collaboration": "Reflexion"}
    thinking3, answer3 = await cot_agent3([taskInfo, thinking1, answer1, thinking2, answer2], cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, assigning stereochemistry, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max):
        feedback3, correct3 = await critic_agent3([taskInfo, thinking3, answer3], "Review the stereochemistry assignment and provide any corrections or validation.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent3.id}, feedback: {feedback3.content}; correct: {correct3.content}")
        if correct3.content == "True":
            break
        thinking3, answer3 = await cot_agent3([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, feedback3], cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, refined stereochemistry, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    cot_instruction4 = "Sub-task 4: Analyze trans-epoxide: determine chirality and count stereoisomers for racemic pair."
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_instruction4, "context": ["subtask_3"], "agent_collaboration": "CoT"}
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, analyzing trans-epoxide, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    cot_sc_instruction5 = "Sub-task 5: Analyze cis-epoxide: determine meso nature and count stereoisomers."
    N5 = self.max_sc
    cot_agents5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N5)]
    possible5 = []
    tmap5 = {}
    amap5 = {}
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": cot_sc_instruction5, "context": ["subtask_3"], "agent_collaboration": "SC_CoT"}
    for i in range(N5):
        t5, a5 = await cot_agents5[i]([taskInfo, thinking3, answer3], cot_sc_instruction5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents5[i].id}, analyzing cis-epoxide, thinking: {t5.content}; answer: {a5.content}")
        possible5.append(a5.content)
        tmap5[a5.content] = t5
        amap5[a5.content] = a5
    a5_content = Counter(possible5).most_common(1)[0][0]
    thinking5 = tmap5[a5_content]
    answer5 = amap5[a5_content]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    cot_reflect_instruction6 = "Sub-task 6: Combine results from subtasks 4 and 5 to list all stereoisomers and give total count."
    cot_agent6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {"subtask_id": "subtask_6", "instruction": cot_reflect_instruction6, "context": ["subtask_4", "subtask_5"], "agent_collaboration": "Reflexion"}
    thinking6, answer6 = await cot_agent6([taskInfo, thinking4, answer4, thinking5, answer5], cot_reflect_instruction6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent6.id}, combining stereoisomers, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max):
        feedback6, correct6 = await critic_agent6([taskInfo, thinking6, answer6], "Review combined stereoisomer list and count accuracy.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent6.id}, feedback: {feedback6.content}; correct: {correct6.content}")
        if correct6.content == "True":
            break
        thinking6, answer6 = await cot_agent6([taskInfo, thinking4, answer4, thinking5, answer5, thinking6, answer6, feedback6], cot_reflect_instruction6, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent6.id}, refined combination, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    cot_sc_instruction7 = "Sub-task 7: Predict peaks in standard HPLC: determine co-elution and separation of stereoisomers."
    N7 = self.max_sc
    cot_agents7 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N7)]
    possible7 = []
    tmap7 = {}
    amap7 = {}
    subtask_desc7 = {"subtask_id": "subtask_7", "instruction": cot_sc_instruction7, "context": ["subtask_6"], "agent_collaboration": "SC_CoT"}
    for i in range(N7):
        t7, a7 = await cot_agents7[i]([taskInfo, thinking6, answer6], cot_sc_instruction7, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents7[i].id}, predicting standard HPLC, thinking: {t7.content}; answer: {a7.content}")
        possible7.append(a7.content)
        tmap7[a7.content] = t7
        amap7[a7.content] = a7
    a7_content = Counter(possible7).most_common(1)[0][0]
    thinking7 = tmap7[a7_content]
    answer7 = amap7[a7_content]
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    cot_sc_instruction8 = "Sub-task 8: Predict peaks in chiral HPLC: determine separation of enantiomers and meso diastereomer."
    N8 = self.max_sc
    cot_agents8 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N8)]
    possible8 = []
    tmap8 = {}
    amap8 = {}
    subtask_desc8 = {"subtask_id": "subtask_8", "instruction": cot_sc_instruction8, "context": ["subtask_6"], "agent_collaboration": "SC_CoT"}
    for i in range(N8):
        t8, a8 = await cot_agents8[i]([taskInfo, thinking6, answer6], cot_sc_instruction8, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents8[i].id}, predicting chiral HPLC, thinking: {t8.content}; answer: {a8.content}")
        possible8.append(a8.content)
        tmap8[a8.content] = t8
        amap8[a8.content] = a8
    a8_content = Counter(possible8).most_common(1)[0][0]
    thinking8 = tmap8[a8_content]
    answer8 = amap8[a8_content]
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])
    debate_instruction9 = "Sub-task 9: Compare predicted peak counts to choices and select the correct answer letter."
    debate_agents9 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max9 = self.max_round
    all_thinking9 = [[] for _ in range(N_max9)]
    all_answer9 = [[] for _ in range(N_max9)]
    subtask_desc9 = {"subtask_id": "subtask_9", "instruction": debate_instruction9, "context": ["subtask_7", "subtask_8"], "agent_collaboration": "Debate"}
    for r in range(N_max9):
        for i, agent in enumerate(debate_agents9):
            if r == 0:
                t9, a9 = await agent([taskInfo, thinking7, answer7, thinking8, answer8], debate_instruction9, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking7, answer7, thinking8, answer8] + all_thinking9[r-1] + all_answer9[r-1]
                t9, a9 = await agent(inputs, debate_instruction9, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {t9.content}; answer: {a9.content}")
            all_thinking9[r].append(t9)
            all_answer9[r].append(a9)
    final_decision_agent9 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking9, answer9 = await final_decision_agent9([taskInfo] + all_thinking9[-1] + all_answer9[-1], "Sub-task 9: Make final decision on correct answer letter.", is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer, logs