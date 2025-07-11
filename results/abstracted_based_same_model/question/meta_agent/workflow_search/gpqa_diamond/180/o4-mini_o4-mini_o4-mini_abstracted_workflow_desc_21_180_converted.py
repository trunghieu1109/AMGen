async def forward_180(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = "Sub-task 1: Identify all neutrino-producing branches in the solar pp-chain reactions (pp-I, pp-II, pp-III, etc.)"
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, identifying branches, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction = "Sub-task 2: Characterize neutrino energy spectrum for each branch identified, specifying energy ranges and relative flux contributions"
    N2 = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_answers2 = []
    thinkingmap2 = {}
    answermap2 = {}
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction, "context": ["user query", "thinking of subtask_1", "answer of subtask_1"], "agent_collaboration": "SC_CoT"}
    for i in range(N2):
        thinking2, answer2 = await cot_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, characterizing spectrum, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers2.append(answer2.content)
        thinkingmap2[answer2.content] = thinking2
        answermap2[answer2.content] = answer2
    answer2_content = Counter(possible_answers2).most_common(1)[0][0]
    thinking2 = thinkingmap2[answer2_content]
    answer2 = answermap2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot_reflect_instruction = "Sub-task 3: Determine which branches produce neutrinos in the 700–900 keV window and note the pp-III branch’s share in that interval"
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    inputs3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_reflect_instruction, "context": ["user query", "thinking of subtask_1", "answer of subtask_1", "thinking of subtask_2", "answer of subtask_2"], "agent_collaboration": "Reflexion"}
    thinking3, answer3 = await cot_agent3(inputs3, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, filtering branches in energy window, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(self.max_round):
        feedback3, correct3 = await critic_agent3([taskInfo, thinking3, answer3], "please review the branch selection and pp-III share calculation and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent3.id}, feedback: {feedback3.content}; correct: {correct3.content}")
        if correct3.content == "True":
            break
        inputs3.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot_agent3(inputs3, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, refining branch selection, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    cot_instruction4 = "Sub-task 4: Account for the 8.5 minute neutrino travel time from Sun to Earth and confirm effect of pp-III stoppage"
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_instruction4, "context": ["user query", "thinking of subtask_1", "answer of subtask_1"], "agent_collaboration": "CoT"}
    thinking4, answer4 = await cot_agent4([taskInfo, thinking1, answer1], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, accounting travel time, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    debate_instruction5 = "Sub-task 5: Model removal of the pp-III branch contribution from the overall solar neutrino spectrum arriving at Earth"
    debate_agents5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N5 = self.max_round
    all_thinking5 = [[] for _ in range(N5)]
    all_answer5 = [[] for _ in range(N5)]
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": debate_instruction5, "context": ["user query", "thinking of subtask_2", "answer of subtask_2", "thinking of subtask_4", "answer of subtask_4"], "agent_collaboration": "Debate"}
    for r in range(N5):
        for i, agent in enumerate(debate_agents5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking2, answer2, thinking4, answer4], debate_instruction5, r, is_sub_task=True)
            else:
                inputs5 = [taskInfo, thinking2, answer2, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(inputs5, debate_instruction5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, modeling removal, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on removed spectrum model.", is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    cot_instruction6 = "Sub-task 6: Generate the modified neutrino energy spectrum at Earth after excluding pp-III branch neutrinos"
    cot_agent6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {"subtask_id": "subtask_6", "instruction": cot_instruction6, "context": ["user query", "thinking of subtask_5", "answer of subtask_5"], "agent_collaboration": "CoT"}
    thinking6, answer6 = await cot_agent6([taskInfo, thinking5, answer5], cot_instruction6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent6.id}, generating modified spectrum, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    cot_instruction7 = "Sub-task 7: Integrate the modified spectrum over 700–800 keV to compute flux for band 1"
    cot_agent7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {"subtask_id": "subtask_7", "instruction": cot_instruction7, "context": ["user query", "thinking of subtask_6", "answer of subtask_6"], "agent_collaboration": "CoT"}
    thinking7, answer7 = await cot_agent7([taskInfo, thinking6, answer6], cot_instruction7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent7.id}, integrating band1, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    cot_instruction8 = "Sub-task 8: Integrate the modified spectrum over 800–900 keV to compute flux for band 2"
    cot_agent8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc8 = {"subtask_id": "subtask_8", "instruction": cot_instruction8, "context": ["user query", "thinking of subtask_6", "answer of subtask_6"], "agent_collaboration": "CoT"}
    thinking8, answer8 = await cot_agent8([taskInfo, thinking6, answer6], cot_instruction8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent8.id}, integrating band2, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])
    cot_sc_instruction9 = "Sub-task 9: Calculate the ratio Flux(band1)/Flux(band2) using the integrated fluxes"
    N9 = self.max_sc
    cot_agents9 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N9)]
    possible_answers9 = []
    thinkingmap9 = {}
    answermap9 = {}
    subtask_desc9 = {"subtask_id": "subtask_9", "instruction": cot_sc_instruction9, "context": ["user query", "thinking of subtask_7", "answer of subtask_7", "thinking of subtask_8", "answer of subtask_8"], "agent_collaboration": "SC_CoT"}
    for i in range(N9):
        thinking9, answer9 = await cot_agents9[i]([taskInfo, thinking7, answer7, thinking8, answer8], cot_sc_instruction9, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents9[i].id}, calculating ratio, thinking: {thinking9.content}; answer: {answer9.content}")
        possible_answers9.append(answer9.content)
        thinkingmap9[answer9.content] = thinking9
        answermap9[answer9.content] = answer9
    answer9_content = Counter(possible_answers9).most_common(1)[0][0]
    thinking9 = thinkingmap9[answer9_content]
    answer9 = answermap9[answer9_content]
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    print("Step 9: ", sub_tasks[-1])
    cot_instruction10 = "Sub-task 10: Compare the computed ratio to the choices and select the matching option (0.1, 10, 1, or 0.01)"
    cot_agent10 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc10 = {"subtask_id": "subtask_10", "instruction": cot_instruction10, "context": ["user query", "thinking of subtask_9", "answer of subtask_9"], "agent_collaboration": "CoT"}
    thinking10, answer10 = await cot_agent10([taskInfo, thinking9, answer9], cot_instruction10, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent10.id}, comparing to choices, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    subtask_desc10['response'] = {"thinking": thinking10, "answer": answer10}
    logs.append(subtask_desc10)
    print("Step 10: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking10, answer10, sub_tasks, agents)
    return final_answer, logs