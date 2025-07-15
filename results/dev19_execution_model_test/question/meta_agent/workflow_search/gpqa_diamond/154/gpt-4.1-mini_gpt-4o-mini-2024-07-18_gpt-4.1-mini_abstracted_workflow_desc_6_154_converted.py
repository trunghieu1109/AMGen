async def forward_154(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction = (
        "Sub-task 1: Extract and verify the given operators Px, Py, and Pz matrices and the state vector; "
        "confirm the basis and normalization of the state vector; summarize all relevant information for further calculations."
    )
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                             model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, analyzing operators and state vector, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    debate_instr_2 = (
        "Sub-task 2: Compute the expectation value <Pz> of the operator Pz in the given state vector. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                    model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_2 = self.max_round
    all_thinking2 = [[] for _ in range(N_max_2)]
    all_answer2 = [[] for _ in range(N_max_2)]
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": debate_instr_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2):
        for i, agent in enumerate(debate_agents_2):
            if r == 0:
                thinking2, answer2 = await agent([taskInfo, thinking1, answer1], 
                                                 debate_instr_2, r, is_sub_task=True)
            else:
                input_infos_2 = [taskInfo, thinking1, answer1] + all_thinking2[r-1] + all_answer2[r-1]
                thinking2, answer2 = await agent(input_infos_2, debate_instr_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, computing <Pz>, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking2[r].append(thinking2)
            all_answer2[r].append(answer2)
    final_decision_instr_2 = (
        "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    )
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                          model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2(
        [taskInfo, thinking1, answer1] + all_thinking2[-1] + all_answer2[-1], 
        "Sub-task 2: Compute <Pz> " + final_decision_instr_2, 
        is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    debate_instr_3 = (
        "Sub-task 3: Compute the expectation value <Pz^2> of the operator Pz squared in the given state vector. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                    model=self.node_model, role=role, temperature=0.5) 
                      for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking3 = [[] for _ in range(N_max_3)]
    all_answer3 = [[] for _ in range(N_max_3)]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instr_3,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking1, answer1], 
                                                 debate_instr_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking1, answer1] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3, answer3 = await agent(input_infos_3, debate_instr_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, computing <Pz^2>, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)
    final_decision_instr_3 = (
        "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    )
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                          model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3(
        [taskInfo, thinking1, answer1] + all_thinking3[-1] + all_answer3[-1], 
        "Sub-task 3: Compute <Pz^2> " + final_decision_instr_3, 
        is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    reflect_inst = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot_reflect_instruction = (
        "Sub-task 4: Calculate the uncertainty ΔPz = sqrt(<Pz^2> - <Pz>^2) using the previously computed expectation values "
        "and interpret the result in terms of ħ. " + reflect_inst
    )
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                             model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                               model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking2, answer2, thinking3, answer3]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_reflect_instruction,
        "context": ["user query", thinking2.content, answer2.content, thinking3.content, answer3.content],
        "agent_collaboration": "Reflexion"
    }
    thinking4, answer4 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, calculating uncertainty ΔPz, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking4, answer4], 
                                              "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", 
                                              i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining uncertainty calculation, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs
