async def forward_95(self, taskInfo):
    from collections import Counter
    import math
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_sc_instruction_1A = "Sub-task 1A: Numerically convert the given angular size θ = 10^-17 degrees into radians to enable precise physical size calculations."
    N1A = self.max_sc
    cot_agents_1A = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1A)]
    possible_answers_1A = []
    thinkingmapping_1A = {}
    answermapping_1A = {}
    subtask_desc1A = {
        "subtask_id": "subtask_1A",
        "instruction": cot_sc_instruction_1A,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N1A):
        thinking1A, answer1A = await cot_agents_1A[i]([taskInfo], cot_sc_instruction_1A, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1A[i].id}, converting angular size to radians, thinking: {thinking1A.content}; answer: {answer1A.content}")
        possible_answers_1A.append(answer1A.content)
        thinkingmapping_1A[answer1A.content] = thinking1A
        answermapping_1A[answer1A.content] = answer1A
    most_common_answer_1A = Counter(possible_answers_1A).most_common(1)[0][0]
    thinking1A = thinkingmapping_1A[most_common_answer_1A]
    answer1A = answermapping_1A[most_common_answer_1A]
    sub_tasks.append(f"Sub-task 1A output: thinking - {thinking1A.content}; answer - {answer1A.content}")
    subtask_desc1A['response'] = {"thinking": thinking1A, "answer": answer1A}
    logs.append(subtask_desc1A)
    print("Step 1A: ", sub_tasks[-1])
    
    cot_sc_instruction_2A = "Sub-task 2A: Convert the distance d = 10^10 parsecs into meters and numerically compute the physical diameter D of the event horizon using D = θ_rad × d."
    N2A = self.max_sc
    cot_agents_2A = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2A)]
    possible_answers_2A = []
    thinkingmapping_2A = {}
    answermapping_2A = {}
    subtask_desc2A = {
        "subtask_id": "subtask_2A",
        "instruction": cot_sc_instruction_2A,
        "context": ["user query", "thinking of subtask 1A", "answer of subtask 1A"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N2A):
        thinking2A, answer2A = await cot_agents_2A[i]([taskInfo, thinking1A, answer1A], cot_sc_instruction_2A, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2A[i].id}, converting distance and calculating physical diameter, thinking: {thinking2A.content}; answer: {answer2A.content}")
        possible_answers_2A.append(answer2A.content)
        thinkingmapping_2A[answer2A.content] = thinking2A
        answermapping_2A[answer2A.content] = answer2A
    most_common_answer_2A = Counter(possible_answers_2A).most_common(1)[0][0]
    thinking2A = thinkingmapping_2A[most_common_answer_2A]
    answer2A = answermapping_2A[most_common_answer_2A]
    sub_tasks.append(f"Sub-task 2A output: thinking - {thinking2A.content}; answer - {answer2A.content}")
    subtask_desc2A['response'] = {"thinking": thinking2A, "answer": answer2A}
    logs.append(subtask_desc2A)
    print("Step 2A: ", sub_tasks[-1])
    
    cot_reflect_instruction_3A = "Sub-task 3A: Calculate the Schwarzschild radius R_s = D/2 numerically and compute the black hole mass M using M = c^2 × R_s / (2G), substituting all constants and previously computed values."
    cot_agent_3A = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3A = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3A = self.max_round
    cot_inputs_3A = [taskInfo, thinking2A, answer2A]
    subtask_desc3A = {
        "subtask_id": "subtask_3A",
        "instruction": cot_reflect_instruction_3A,
        "context": ["user query", "thinking of subtask 2A", "answer of subtask 2A"],
        "agent_collaboration": "Reflexion"
    }
    thinking3A, answer3A = await cot_agent_3A(cot_inputs_3A, cot_reflect_instruction_3A, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3A.id}, calculating Schwarzschild radius and mass, thinking: {thinking3A.content}; answer: {answer3A.content}")
    for i in range(N_max_3A):
        feedback, correct = await critic_agent_3A([taskInfo, thinking3A, answer3A], "please review the Schwarzschild radius and mass calculation for correctness and numeric accuracy.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3A.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3A.extend([thinking3A, answer3A, feedback])
        thinking3A, answer3A = await cot_agent_3A(cot_inputs_3A, cot_reflect_instruction_3A, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3A.id}, refining Schwarzschild radius and mass calculation, thinking: {thinking3A.content}; answer: {answer3A.content}")
    sub_tasks.append(f"Sub-task 3A output: thinking - {thinking3A.content}; answer - {answer3A.content}")
    subtask_desc3A['response'] = {"thinking": thinking3A, "answer": answer3A}
    logs.append(subtask_desc3A)
    print("Step 3A: ", sub_tasks[-1])
    
    debate_instruction_4A = "Sub-task 4A: Compute the event horizon area A = 4π R_s^2 numerically and evaluate the Bekenstein-Hawking entropy S = (k c^3 A) / (4 G ħ) using the calculated radius and physical constants."
    debate_agents_4A = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4A = self.max_round
    all_thinking4A = [[] for _ in range(N_max_4A)]
    all_answer4A = [[] for _ in range(N_max_4A)]
    subtask_desc4A = {
        "subtask_id": "subtask_4A",
        "instruction": debate_instruction_4A,
        "context": ["user query", "thinking of subtask 3A", "answer of subtask 3A"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4A):
        for i, agent in enumerate(debate_agents_4A):
            if r == 0:
                thinking4A, answer4A = await agent([taskInfo, thinking3A, answer3A], debate_instruction_4A, r, is_sub_task=True)
            else:
                input_infos_4A = [taskInfo, thinking3A, answer3A] + all_thinking4A[r-1] + all_answer4A[r-1]
                thinking4A, answer4A = await agent(input_infos_4A, debate_instruction_4A, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, calculating entropy, thinking: {thinking4A.content}; answer: {answer4A.content}")
            all_thinking4A[r].append(thinking4A)
            all_answer4A[r].append(answer4A)
    final_decision_agent_4A = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4A, answer4A = await final_decision_agent_4A([taskInfo] + all_thinking4A[-1] + all_answer4A[-1], "Sub-task 4A: Make final decision on the entropy calculation.", is_sub_task=True)
    agents.append(f"Final Decision agent on entropy calculation, thinking: {thinking4A.content}; answer: {answer4A.content}")
    sub_tasks.append(f"Sub-task 4A output: thinking - {thinking4A.content}; answer - {answer4A.content}")
    subtask_desc4A['response'] = {"thinking": thinking4A, "answer": answer4A}
    logs.append(subtask_desc4A)
    print("Step 4A: ", sub_tasks[-1])
    
    cot_sc_instruction_5A = "Sub-task 5A: Determine the order of magnitude of the computed entropy S, compare it quantitatively with the provided multiple-choice options, and select the correct answer based on numeric evaluation."
    N5A = self.max_sc
    cot_agents_5A = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N5A)]
    possible_answers_5A = []
    thinkingmapping_5A = {}
    answermapping_5A = {}
    subtask_desc5A = {
        "subtask_id": "subtask_5A",
        "instruction": cot_sc_instruction_5A,
        "context": ["user query", "thinking of subtask 4A", "answer of subtask 4A"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N5A):
        thinking5A, answer5A = await cot_agents_5A[i]([taskInfo, thinking4A, answer4A], cot_sc_instruction_5A, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5A[i].id}, determining entropy order of magnitude and selecting answer, thinking: {thinking5A.content}; answer: {answer5A.content}")
        possible_answers_5A.append(answer5A.content)
        thinkingmapping_5A[answer5A.content] = thinking5A
        answermapping_5A[answer5A.content] = answer5A
    most_common_answer_5A = Counter(possible_answers_5A).most_common(1)[0][0]
    thinking5A = thinkingmapping_5A[most_common_answer_5A]
    answer5A = answermapping_5A[most_common_answer_5A]
    sub_tasks.append(f"Sub-task 5A output: thinking - {thinking5A.content}; answer - {answer5A.content}")
    subtask_desc5A['response'] = {"thinking": thinking5A, "answer": answer5A}
    logs.append(subtask_desc5A)
    print("Step 5A: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking5A, answer5A, sub_tasks, agents)
    return final_answer, logs
