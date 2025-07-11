async def forward_14(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = "Sub-task 1: Parameterize a general point (x,y) on the hyperbola x^2/20 - y^2/24 = 1 using a real parameter t."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, parameterizing hyperbola point, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    cot_sc_instruction2 = "Sub-task 2: Express the four vertices A, B, C, D of the rhombus in terms of parameters t and u, and use the condition that the diagonals intersect at the origin to show C = -A and D = -B."
    N2 = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_answers2 = []
    thinkingmapping2 = {}
    answermapping2 = {}
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction2, "context": ["user query", "thinking of subtask 1", "answer of subtask 1"], "agent_collaboration": "SC_CoT"}
    for i in range(N2):
        thinking2, answer2 = await cot_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, expressing vertices and diagonal condition, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers2.append(answer2.content)
        thinkingmapping2[answer2.content] = thinking2
        answermapping2[answer2.content] = answer2
    answer2_content = Counter(possible_answers2).most_common(1)[0][0]
    thinking2 = thinkingmapping2[answer2_content]
    answer2 = answermapping2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    reflect_instruction3 = "Sub-task 3: Formulate the rhombus side‐length equality |A - B| = |A + B| to derive a relation between u and t."
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N3 = self.max_round
    cot_inputs3 = [taskInfo, thinking2, answer2]
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": reflect_instruction3, "context": ["user query", "thinking of subtask 2", "answer of subtask 2"], "agent_collaboration": "Reflexion"}
    thinking3, answer3 = await cot_agent3(cot_inputs3, reflect_instruction3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, deriving side-length relation, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N3):
        feedback3, correct3 = await critic_agent3([taskInfo, thinking3, answer3], "please review the relation between u and t and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent3.id}, providing feedback, thinking: {feedback3.content}; answer: {correct3.content}")
        if correct3.content == "True":
            break
        cot_inputs3.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot_agent3(cot_inputs3, reflect_instruction3, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, refining relation, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    cot_instruction4 = "Sub-task 4: Restate the optimization goal in your own words, clarifying that the problem seeks the infimum of BD^2 over all such rhombi."
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_instruction4, "context": ["user query", "thinking of subtask 3", "answer of subtask 3"], "agent_collaboration": "CoT"}
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, clarifying optimization goal, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    cot_instruction5 = "Sub-task 5: Compute the squared diagonal length BD^2 = |B - D|^2 in terms of t and u."
    cot_agent5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": cot_instruction5, "context": ["user query", "thinking of subtask 2", "answer of subtask 2"], "agent_collaboration": "CoT"}
    thinking5, answer5 = await cot_agent5([taskInfo, thinking2, answer2], cot_instruction5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent5.id}, computing BD^2 expression, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    sc_instruction6 = "Sub-task 6: Use the relation from subtask_3 to eliminate u and express BD^2 as a single-variable function f(t)."
    N6 = self.max_sc
    cot_agents6 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N6)]
    possible_answers6 = []
    thinkingmapping6 = {}
    answermapping6 = {}
    subtask_desc6 = {"subtask_id": "subtask_6", "instruction": sc_instruction6, "context": ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 5", "answer of subtask 5"], "agent_collaboration": "SC_CoT"}
    for i in range(N6):
        thinking6, answer6 = await cot_agents6[i]([taskInfo, thinking3, answer3, thinking5, answer5], sc_instruction6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents6[i].id}, eliminating u to get f(t), thinking: {thinking6.content}; answer: {answer6.content}")
        possible_answers6.append(answer6.content)
        thinkingmapping6[answer6.content] = thinking6
        answermapping6[answer6.content] = answer6
    answer6_content = Counter(possible_answers6).most_common(1)[0][0]
    thinking6 = thinkingmapping6[answer6_content]
    answer6 = answermapping6[answer6_content]
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    reflect_instruction7 = "Sub-task 7: Determine the range of f(t) by evaluating its behavior as t approaches the domain endpoints and infinity."
    cot_agent7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N7 = self.max_round
    cot_inputs7 = [taskInfo, thinking6, answer6]
    subtask_desc7 = {"subtask_id": "subtask_7", "instruction": reflect_instruction7, "context": ["user query", "thinking of subtask 6", "answer of subtask 6"], "agent_collaboration": "Reflexion"}
    thinking7, answer7 = await cot_agent7(cot_inputs7, reflect_instruction7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent7.id}, evaluating range of f(t), thinking: {thinking7.content}; answer: {answer7.content}")
    for i in range(N7):
        feedback7, correct7 = await critic_agent7([taskInfo, thinking7, answer7], "please review the range evaluation of f(t) and ensure endpoints and infinity are considered.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent7.id}, providing feedback, thinking: {feedback7.content}; answer: {correct7.content}")
        if correct7.content == "True":
            break
        cot_inputs7.extend([thinking7, answer7, feedback7])
        thinking7, answer7 = await cot_agent7(cot_inputs7, reflect_instruction7, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent7.id}, refining range evaluation, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    cot_instruction8 = "Sub-task 8: Based on the clarified goal from subtask_4, select the infimum of the range of f(t)."
    cot_agent8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc8 = {"subtask_id": "subtask_8", "instruction": cot_instruction8, "context": ["user query", "thinking of subtask 4", "answer of subtask 4", "thinking of subtask 7", "answer of subtask 7"], "agent_collaboration": "CoT"}
    thinking8, answer8 = await cot_agent8([taskInfo, thinking4, answer4, thinking7, answer7], cot_instruction8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent8.id}, selecting infimum of f(t), thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print("Step 8: ", sub_tasks[-1])
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    reflect_instruction9 = "Sub-task 9: Validate that the chosen value is strictly less than BD^2 for every rhombus satisfying the constraints."
    cot_agent9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent9 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N9 = self.max_round
    cot_inputs9 = [taskInfo, thinking8, answer8]
    subtask_desc9 = {"subtask_id": "subtask_9", "instruction": reflect_instruction9, "context": ["user query", "thinking of subtask 8", "answer of subtask 8"], "agent_collaboration": "Reflexion"}
    thinking9, answer9 = await cot_agent9(cot_inputs9, reflect_instruction9, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent9.id}, validating infimum, thinking: {thinking9.content}; answer: {answer9.content}")
    for i in range(N9):
        feedback9, correct9 = await critic_agent9([taskInfo, thinking9, answer9], "please check that the infimum is less than BD^2 for all valid rhombi.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent9.id}, providing feedback, thinking: {feedback9.content}; answer: {correct9.content}")
        if correct9.content == "True":
            break
        cot_inputs9.extend([thinking9, answer9, feedback9])
        thinking9, answer9 = await cot_agent9(cot_inputs9, reflect_instruction9, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent9.id}, refining validation, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    print("Step 9: ", sub_tasks[-1])
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    cot_instruction10 = "Sub-task 10: Return the integer value of the infimum, strictly following the output requirement (only the integer)."
    cot_agent10 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc10 = {"subtask_id": "subtask_10", "instruction": cot_instruction10, "context": ["user query", "answer of subtask 8"], "agent_collaboration": "CoT"}
    thinking10, answer10 = await cot_agent10([taskInfo, answer8], cot_instruction10, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent10.id}, returning integer infimum, thinking: {thinking10.content}; answer: {answer10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking10.content}; answer - {answer10.content}")
    print("Step 10: ", sub_tasks[-1])
    subtask_desc10['response'] = {"thinking": thinking10, "answer": answer10}
    logs.append(subtask_desc10)
    final_answer = await self.make_final_answer(thinking10, answer10, sub_tasks, agents)
    return final_answer, logs