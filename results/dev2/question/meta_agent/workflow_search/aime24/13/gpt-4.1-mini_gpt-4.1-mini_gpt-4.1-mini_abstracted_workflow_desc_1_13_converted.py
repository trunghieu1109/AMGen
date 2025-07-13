async def forward_13(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Rigorously analyze the geometric configuration of the chain of tangent circles inside triangle ABC. "
        "Derive exact relationships between the inradius R, angle at vertex B (theta), number of circles n, their radius r, and the triangle sides. "
        "Explicitly formulate the system of equations from the tangency conditions without assuming proportionality or fixed parameters. "
        "Provide detailed derivations and justifications for all geometric constraints."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, rigorous geometric derivation, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_2 = (
        "Sub-task 2: Using the geometric relationships and system of equations derived in Sub-task 1, "
        "formulate the specific equations for the two given configurations: (n=8, r=34) and (n=2024, r=1). "
        "Set up a system of two equations with two unknowns (e.g., angle theta and inradius R or related dimension) that fully capture the problem constraints. "
        "Provide detailed algebraic expressions and justifications."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", thinking1, answer1],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, formulating system of equations, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    debate_instruction_3 = (
        "Sub-task 3: Engage in a debate where one agent proposes algebraic elimination and solution steps for the system of equations from Sub-task 2, "
        "and another agent critically evaluates each step, questioning assumptions and verifying correctness. "
        "Resolve contradictions or flawed assumptions through interactive discussion to ensure a consistent and justified solution path. "
        "Provide detailed reasoning and final agreed system ready for solving."
    )
    debate_agents_3 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role="Proposer", temperature=0.5),
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role="Critic", temperature=0.5)
    ]
    N_max_3 = self.max_round
    all_thinking3 = [[] for _ in range(N_max_3)]
    all_answer3 = [[] for _ in range(N_max_3)]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instruction_3,
        "context": ["user query", thinking1, answer1, thinking2, answer2],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2, answer2], debate_instruction_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3, answer3 = await agent(input_infos_3, debate_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, role {agent.role}, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + all_thinking3[-1] + all_answer3[-1], "Sub-task 3: Make final decision on verified system of equations.", is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing system of equations, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_4 = (
        "Sub-task 4: Solve the verified system of equations from Sub-task 3 to find exact values of the inradius R and angle theta (or other relevant parameters). "
        "Express the inradius as a reduced fraction m/n with relatively prime positive integers. "
        "Provide detailed algebraic steps and justifications."
    )
    N = self.max_sc
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", thinking1, answer1, thinking2, answer2, thinking3, answer3],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, solving system for inradius, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    answer4_content = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[answer4_content]
    answer4 = answermapping_4[answer4_content]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_reflect_instruction_5 = (
        "Sub-task 5: Perform rigorous verification of the solution from Sub-task 4 by checking consistency with both given configurations (8 circles of radius 34 and 2024 circles of radius 1). "
        "Explicitly verify all assumptions, explore alternative solution paths if contradictions arise, confirm the fraction is in lowest terms, and compute the sum m+n as required. "
        "Provide comprehensive justification of the solution's correctness."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    cot_inputs_5 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_reflect_instruction_5,
        "context": ["user query", thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4],
        "agent_collaboration": "Reflexion"
    }
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, verifying and refining solution, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max_5):
        feedback, correct = await critic_agent_5([taskInfo, thinking5, answer5], "please review the verification and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, feedback on verification, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip().lower() == "true":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, cot_reflect_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining verification, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
