async def forward_145(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1a = "Sub-task 1a: Identify and analyze the structural features of 5-fluorocyclopenta-1,3-diene relevant to its reactivity, including the position of the fluorine substituent and the diene system."
    cot_agent_1a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1a = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_1a, answer_1a = await cot_agent_1a([taskInfo], cot_instruction_1a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1a.id}, analyzing 5-fluorocyclopenta-1,3-diene, thinking: {thinking_1a.content}; answer: {answer_1a.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1a.content}; answer - {answer_1a.content}")
    subtask_desc_1a['response'] = {"thinking": thinking_1a, "answer": answer_1a}
    logs.append(subtask_desc_1a)
    print("Step 1: ", sub_tasks[-1])
    
    cot_instruction_1b = "Sub-task 1b: Identify and analyze the structural features and reactive sites of maleic anhydride relevant to its role as a dienophile in Diels-Alder reactions."
    cot_agent_1b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1b = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_1b,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_1b, answer_1b = await cot_agent_1b([taskInfo], cot_instruction_1b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1b.id}, analyzing maleic anhydride, thinking: {thinking_1b.content}; answer: {answer_1b.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1b.content}; answer - {answer_1b.content}")
    subtask_desc_1b['response'] = {"thinking": thinking_1b, "answer": answer_1b}
    logs.append(subtask_desc_1b)
    print("Step 2: ", sub_tasks[-1])
    
    cot_sc_instruction_2a = "Sub-task 3: Determine the type of reaction expected between 5-fluorocyclopenta-1,3-diene and maleic anhydride, focusing on the Diels-Alder cycloaddition mechanism, based on the outputs from Sub-task 1 and Sub-task 2."
    N = self.max_sc
    cot_agents_2a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2a = []
    thinkingmapping_2a = {}
    answermapping_2a = {}
    subtask_desc_2a = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_2a,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking_2a, answer_2a = await cot_agents_2a[i]([taskInfo, thinking_1a, answer_1a, thinking_1b, answer_1b], cot_sc_instruction_2a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2a[i].id}, determining reaction type, thinking: {thinking_2a.content}; answer: {answer_2a.content}")
        possible_answers_2a.append(answer_2a.content)
        thinkingmapping_2a[answer_2a.content] = thinking_2a
        answermapping_2a[answer_2a.content] = answer_2a
    most_common_answer_2a = Counter(possible_answers_2a).most_common(1)[0][0]
    thinking_2a = thinkingmapping_2a[most_common_answer_2a]
    answer_2a = answermapping_2a[most_common_answer_2a]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_2a.content}; answer - {answer_2a.content}")
    subtask_desc_2a['response'] = {"thinking": thinking_2a, "answer": answer_2a}
    logs.append(subtask_desc_2a)
    print("Step 3: ", sub_tasks[-1])
    
    cot_reflect_instruction_2b = "Sub-task 4: Predict the regio- and stereochemical outcome of the Diels-Alder reaction between 5-fluorocyclopenta-1,3-diene and maleic anhydride, considering electronic and steric effects of the fluorine substituent, based on the output of Sub-task 3."
    cot_agent_2b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2b = self.max_round
    cot_inputs_2b = [taskInfo, thinking_1a, answer_1a, thinking_1b, answer_1b, thinking_2a, answer_2a]
    thinking_2b, answer_2b = await cot_agent_2b(cot_inputs_2b, cot_reflect_instruction_2b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2b.id}, predicting regio- and stereochemistry, thinking: {thinking_2b.content}; answer: {answer_2b.content}")
    for i in range(N_max_2b):
        feedback_2b, correct_2b = await critic_agent_2b([taskInfo, thinking_2b, answer_2b], "please review the regio- and stereochemical prediction and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2b.id}, providing feedback, thinking: {feedback_2b.content}; answer: {correct_2b.content}")
        if correct_2b.content == "True":
            break
        cot_inputs_2b.extend([thinking_2b, answer_2b, feedback_2b])
        thinking_2b, answer_2b = await cot_agent_2b(cot_inputs_2b, cot_reflect_instruction_2b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2b.id}, refining prediction, thinking: {thinking_2b.content}; answer: {answer_2b.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_2b.content}; answer - {answer_2b.content}")
    subtask_desc_2b = {
        "subtask_id": "subtask_4",
        "instruction": cot_reflect_instruction_2b,
        "context": ["user query", "thinking and answer of subtask 3"],
        "agent_collaboration": "Reflexion"
    }
    subtask_desc_2b['response'] = {"thinking": thinking_2b, "answer": answer_2b}
    logs.append(subtask_desc_2b)
    print("Step 4: ", sub_tasks[-1])
    
    debate_instruction_3a = "Sub-task 5: Assign the absolute stereochemistry (R/S) and relative stereochemistry of the major Diels-Alder adduct based on the predicted stereochemical outcome from Sub-task 4."
    debate_agents_3a = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3a = self.max_round
    all_thinking_3a = [[] for _ in range(N_max_3a)]
    all_answer_3a = [[] for _ in range(N_max_3a)]
    subtask_desc_3a = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_3a,
        "context": ["user query", "thinking and answer of subtask 4"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3a):
        for i, agent in enumerate(debate_agents_3a):
            if r == 0:
                thinking_3a, answer_3a = await agent([taskInfo, thinking_2b, answer_2b], debate_instruction_3a, r, is_sub_task=True)
            else:
                input_infos_3a = [taskInfo, thinking_2b, answer_2b] + all_thinking_3a[r-1] + all_answer_3a[r-1]
                thinking_3a, answer_3a = await agent(input_infos_3a, debate_instruction_3a, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, assigning stereochemistry, thinking: {thinking_3a.content}; answer: {answer_3a.content}")
            all_thinking_3a[r].append(thinking_3a)
            all_answer_3a[r].append(answer_3a)
    final_decision_agent_3a = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3a, answer_3a = await final_decision_agent_3a([taskInfo] + all_thinking_3a[-1] + all_answer_3a[-1], "Sub-task 5: Make final decision on stereochemical assignment.", is_sub_task=True)
    agents.append(f"Final Decision agent, assigning stereochemistry, thinking: {thinking_3a.content}; answer: {answer_3a.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_3a.content}; answer - {answer_3a.content}")
    subtask_desc_3a['response'] = {"thinking": thinking_3a, "answer": answer_3a}
    logs.append(subtask_desc_3a)
    print("Step 5: ", sub_tasks[-1])
    
    debate_instruction_3b = "Sub-task 6: Match the predicted stereochemical structure of the major product with the given multiple-choice options to identify the correct answer."
    debate_agents_3b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3b = self.max_round
    all_thinking_3b = [[] for _ in range(N_max_3b)]
    all_answer_3b = [[] for _ in range(N_max_3b)]
    subtask_desc_3b = {
        "subtask_id": "subtask_6",
        "instruction": debate_instruction_3b,
        "context": ["user query", "thinking and answer of subtask 5", "multiple choice options"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3b):
        for i, agent in enumerate(debate_agents_3b):
            if r == 0:
                thinking_3b, answer_3b = await agent([taskInfo, thinking_3a, answer_3a], debate_instruction_3b, r, is_sub_task=True)
            else:
                input_infos_3b = [taskInfo, thinking_3a, answer_3a] + all_thinking_3b[r-1] + all_answer_3b[r-1]
                thinking_3b, answer_3b = await agent(input_infos_3b, debate_instruction_3b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, matching product with options, thinking: {thinking_3b.content}; answer: {answer_3b.content}")
            all_thinking_3b[r].append(thinking_3b)
            all_answer_3b[r].append(answer_3b)
    final_decision_agent_3b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3b, answer_3b = await final_decision_agent_3b([taskInfo] + all_thinking_3b[-1] + all_answer_3b[-1], "Sub-task 6: Make final decision on the correct major product choice.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting major product, thinking: {thinking_3b.content}; answer: {answer_3b.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking_3b.content}; answer - {answer_3b.content}")
    subtask_desc_3b['response'] = {"thinking": thinking_3b, "answer": answer_3b}
    logs.append(subtask_desc_3b)
    print("Step 6: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking_3b, answer_3b, sub_tasks, agents)
    return final_answer, logs
