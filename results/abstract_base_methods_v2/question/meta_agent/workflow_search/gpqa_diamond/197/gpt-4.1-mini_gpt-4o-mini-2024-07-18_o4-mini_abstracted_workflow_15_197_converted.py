async def forward_197(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = "Sub-task 1: Identify and list all cobalt(II) thiocyanato complexes formed in the solution along with their overall stability constants (β1=9, β2=40, β3=63, β4=16) as given in the problem."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying complexes and stability constants, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_2 = "Sub-task 2: Write the mass balance equation for total cobalt concentration, expressing total cobalt (c(Co) = 10^-2 M) as the sum of free Co(II) ion concentration and the concentrations of all cobalt(II) thiocyanato complexes."
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, writing mass balance equation, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_3 = "Sub-task 3: Express the concentration of each cobalt(II) thiocyanato complex in terms of free Co(II) ion concentration, free SCN- concentration (0.1 M), and the respective overall stability constants (β values)."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, expressing complex concentrations, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_4a = "Sub-task 4a: Formulate the numeric polynomial mass balance equation by substituting known values (total Co concentration = 0.01 M, SCN- concentration = 0.1 M, β1=9, β2=40, β3=63, β4=16) and expressing all complex concentrations in terms of free Co(II) concentration."
    cot_agent_4a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4a = {
        "subtask_id": "subtask_4a",
        "instruction": cot_instruction_4a,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "CoT"
    }
    thinking4a, answer4a = await cot_agent_4a([taskInfo, thinking3, answer3], cot_instruction_4a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4a.id}, formulating polynomial mass balance equation, thinking: {thinking4a.content}; answer: {answer4a.content}")
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking4a.content}; answer - {answer4a.content}")
    subtask_desc4a['response'] = {"thinking": thinking4a, "answer": answer4a}
    logs.append(subtask_desc4a)
    print("Step 4a: ", sub_tasks[-1])

    cot_instruction_4b = "Sub-task 4b: Solve the polynomial mass balance equation numerically to find the free Co(II) ion concentration, using iterative or root-finding methods with detailed intermediate calculations. Generate multiple independent solution attempts for self-consistency."
    N = self.max_sc
    cot_agents_4b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4b = []
    thinkingmapping_4b = {}
    answermapping_4b = {}
    subtask_desc4b = {
        "subtask_id": "subtask_4b",
        "instruction": cot_instruction_4b,
        "context": ["user query", "thinking of subtask 4a", "answer of subtask 4a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4b, answer4b = await cot_agents_4b[i]([taskInfo, thinking4a, answer4a], cot_instruction_4b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4b[i].id}, solving polynomial for free Co(II), thinking: {thinking4b.content}; answer: {answer4b.content}")
        possible_answers_4b.append(answer4b.content)
        thinkingmapping_4b[answer4b.content] = thinking4b
        answermapping_4b[answer4b.content] = answer4b
    answer4b_content = Counter(possible_answers_4b).most_common(1)[0][0]
    thinking4b = thinkingmapping_4b[answer4b_content]
    answer4b = answermapping_4b[answer4b_content]
    sub_tasks.append(f"Sub-task 4b output: thinking - {thinking4b.content}; answer - {answer4b.content}")
    subtask_desc4b['response'] = {"thinking": thinking4b, "answer": answer4b}
    logs.append(subtask_desc4b)
    print("Step 4b: ", sub_tasks[-1])

    cot_instruction_4c = "Sub-task 4c: Verify the obtained free Co(II) concentration by back-substitution into the mass balance equation to confirm consistency and accuracy."
    cot_agent_4c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4c = {
        "subtask_id": "subtask_4c",
        "instruction": cot_instruction_4c,
        "context": ["user query", "thinking of subtask 4b", "answer of subtask 4b", "thinking of subtask 4a", "answer of subtask 4a"],
        "agent_collaboration": "Reflexion"
    }
    cot_inputs_4c = [taskInfo, thinking4b, answer4b, thinking4a, answer4a]
    thinking4c, answer4c = await cot_agent_4c(cot_inputs_4c, cot_instruction_4c, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4c.id}, verifying free Co(II) concentration, thinking: {thinking4c.content}; answer: {answer4c.content}")
    critic_agent_4c = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4c = self.max_round
    for i in range(N_max_4c):
        feedback, correct = await critic_agent_4c([taskInfo, thinking4c, answer4c], "Please review the verification of free Co(II) concentration and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4c.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4c.extend([thinking4c, answer4c, feedback])
        thinking4c, answer4c = await cot_agent_4c(cot_inputs_4c, cot_instruction_4c, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4c.id}, refining verification, thinking: {thinking4c.content}; answer: {answer4c.content}")
    sub_tasks.append(f"Sub-task 4c output: thinking - {thinking4c.content}; answer - {answer4c.content}")
    subtask_desc4c['response'] = {"thinking": thinking4c, "answer": answer4c}
    logs.append(subtask_desc4c)
    print("Step 4c: ", sub_tasks[-1])

    cot_instruction_5 = "Sub-task 5: Calculate the concentration of the dithiocyanato cobalt(II) complex (the complex with two SCN- ligands) using the verified free Co(II) concentration and the corresponding stability constant (β2=40)."
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", "thinking of subtask 4c", "answer of subtask 4c"],
        "agent_collaboration": "CoT"
    }
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4c, answer4c], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, calculating dithiocyanato complex concentration, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot_instruction_6 = "Sub-task 6: Calculate the total concentration of all cobalt-containing species by numerically summing the free Co(II) ion concentration and the concentrations of all cobalt(II) thiocyanato complexes using the verified free Co(II) concentration."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", "thinking of subtask 4c", "answer of subtask 4c", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking4c, answer4c, thinking3, answer3], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, calculating total cobalt species concentration, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    cot_instruction_7 = "Sub-task 7: Calculate the percentage of the blue dithiocyanato cobalt(II) complex relative to the total cobalt-containing species concentration using the numeric concentrations obtained."
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_7,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5", "thinking of subtask 6", "answer of subtask 6"],
        "agent_collaboration": "CoT"
    }
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking5, answer5, thinking6, answer6], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, calculating percentage of dithiocyanato complex, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    debate_instruction_8a = "Sub-task 8a: One agent proposes the final percentage calculation result based on subtask_7 output."
    debate_agent_8a = LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=self.debate_role[0], temperature=0.5)
    subtask_desc8a = {
        "subtask_id": "subtask_8a",
        "instruction": debate_instruction_8a,
        "context": ["user query", "thinking of subtask 7", "answer of subtask 7"],
        "agent_collaboration": "Debate"
    }
    thinking8a, answer8a = await debate_agent_8a([taskInfo, thinking7, answer7], debate_instruction_8a, is_sub_task=True)
    agents.append(f"Debate agent {debate_agent_8a.id}, proposing final percentage, thinking: {thinking8a.content}; answer: {answer8a.content}")
    sub_tasks.append(f"Sub-task 8a output: thinking - {thinking8a.content}; answer - {answer8a.content}")
    subtask_desc8a['response'] = {"thinking": thinking8a, "answer": answer8a}
    logs.append(subtask_desc8a)
    print("Step 8a: ", sub_tasks[-1])

    debate_instruction_8b = "Sub-task 8b: Another agent independently verifies the final percentage by recomputing total and ratio to ensure accuracy and consistency."
    debate_agent_8b = LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=self.debate_role[1], temperature=0.5)
    subtask_desc8b = {
        "subtask_id": "subtask_8b",
        "instruction": debate_instruction_8b,
        "context": ["user query", "thinking of subtask 7", "answer of subtask 7"],
        "agent_collaboration": "Debate"
    }
    thinking8b, answer8b = await debate_agent_8b([taskInfo, thinking7, answer7], debate_instruction_8b, is_sub_task=True)
    agents.append(f"Debate agent {debate_agent_8b.id}, verifying final percentage, thinking: {thinking8b.content}; answer: {answer8b.content}")
    sub_tasks.append(f"Sub-task 8b output: thinking - {thinking8b.content}; answer - {answer8b.content}")
    subtask_desc8b['response'] = {"thinking": thinking8b, "answer": answer8b}
    logs.append(subtask_desc8b)
    print("Step 8b: ", sub_tasks[-1])

    debate_instruction_8c = "Sub-task 8c: Compare the verified calculated percentage with the given multiple-choice options and select the correct answer choice (A, B, C, or D)."
    debate_agents_8c = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_8c = self.max_round
    all_thinking8c = [[] for _ in range(N_max_8c)]
    all_answer8c = [[] for _ in range(N_max_8c)]
    subtask_desc8c = {
        "subtask_id": "subtask_8c",
        "instruction": debate_instruction_8c,
        "context": ["user query", "thinking of subtask 8a", "answer of subtask 8a", "thinking of subtask 8b", "answer of subtask 8b"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_8c):
        for i, agent in enumerate(debate_agents_8c):
            if r == 0:
                thinking8c, answer8c = await agent([taskInfo, thinking8a, answer8a, thinking8b, answer8b], debate_instruction_8c, r, is_sub_task=True)
            else:
                input_infos_8c = [taskInfo, thinking8a, answer8a, thinking8b, answer8b] + all_thinking8c[r-1] + all_answer8c[r-1]
                thinking8c, answer8c = await agent(input_infos_8c, debate_instruction_8c, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting correct answer choice, thinking: {thinking8c.content}; answer: {answer8c.content}")
            all_thinking8c[r].append(thinking8c)
            all_answer8c[r].append(answer8c)
    final_decision_agent_8c = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking8c, answer8c = await final_decision_agent_8c([taskInfo] + all_thinking8c[-1] + all_answer8c[-1], "Sub-task 8c: Make final decision on the correct answer choice.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting final answer choice, thinking: {thinking8c.content}; answer: {answer8c.content}")
    sub_tasks.append(f"Sub-task 8c output: thinking - {thinking8c.content}; answer - {answer8c.content}")
    subtask_desc8c['response'] = {"thinking": thinking8c, "answer": answer8c}
    logs.append(subtask_desc8c)
    print("Step 8c: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking8c, answer8c, sub_tasks, agents)
    return final_answer, logs
