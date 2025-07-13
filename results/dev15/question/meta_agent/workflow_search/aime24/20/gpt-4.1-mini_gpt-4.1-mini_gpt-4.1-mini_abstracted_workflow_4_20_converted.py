async def forward_20(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Validate and clarify mathematical formulation and digit constraints
    cot_instruction_0_1 = (
        "Sub-task 1: Validate and clearly restate the mathematical relationship defining b-eautiful integers, "
        "ensuring the equation (x + y)^2 = x*b + y holds under the digit constraints 1 ≤ x ≤ b-1 and 0 ≤ y ≤ b-1, "
        "and confirm that n = x*b + y is a two-digit number in base b with leading digit x nonzero. Avoid assuming any properties without explicit justification."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, validating mathematical relationship, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Stage 0 Subtask 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0.1: ", sub_tasks[-1])

    cot_instruction_0_2 = (
        "Sub-task 2: Clarify and confirm all digit constraints and assumptions, including that the leading digit x must be at least 1, "
        "the digit y is between 0 and b-1, and that n must be a perfect square since s = sqrt(n) is integer. "
        "Explicitly state these constraints and avoid ambiguity."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"Reflexion agent {cot_agent_0_2.id}, clarifying digit constraints, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Stage 0 Subtask 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0.2: ", sub_tasks[-1])

    # Stage 1: Enumerate b-eautiful integers for bases starting from 2 upwards
    cot_sc_instruction_1_1 = (
        "Sub-task 1: For a fixed base b, generate all possible digit pairs (x,y) with 1 ≤ x ≤ b-1 and 0 ≤ y ≤ b-1, "
        "and filter those pairs that satisfy the equation (x + y)^2 = x*b + y, thereby identifying all b-eautiful integers for that base. "
        "Output the full list of valid (x,y) pairs and corresponding n values explicitly."
    )
    N_sc = self.max_sc
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_2.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    for i in range(N_sc):
        thinking_1_1, answer_1_1 = await cot_agents_1_1[i]([taskInfo, thinking_0_2], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, enumerating b-eautiful integers for fixed b, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
        possible_answers_1_1.append(answer_1_1)
        possible_thinkings_1_1.append(thinking_1_1)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + possible_thinkings_1_1, "Sub-task 1: Synthesize and choose the most consistent enumeration of b-eautiful integers for fixed b.", is_sub_task=True)
    sub_tasks.append(f"Stage 1 Subtask 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    cot_sc_instruction_1_2 = (
        "Sub-task 2: Implement a systematic enumeration process for bases b starting from 2 upwards, "
        "applying the generation and filtering process from Sub-task 1 for each base. For each base, count the number of b-eautiful integers and output a detailed table or list of (b, count) pairs. "
        "Continue enumeration until a base with more than 10 b-eautiful integers is found or until b reaches at least 25. "
        "Ensure all enumeration outputs are explicit and verifiable."
    )
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    for i in range(N_sc):
        thinking_1_2, answer_1_2 = await cot_agents_1_2[i]([taskInfo, thinking_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, enumerating b-eautiful integers across bases, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
        possible_answers_1_2.append(answer_1_2)
        possible_thinkings_1_2.append(thinking_1_2)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_thinkings_1_2, "Sub-task 2: Synthesize and choose the most consistent enumeration results across bases.", is_sub_task=True)
    sub_tasks.append(f"Stage 1 Subtask 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    # Stage 1.5: Verification and critical analysis of enumeration results
    debate_instruction_1_5 = (
        "Sub-task 3: Critically analyze and verify the enumeration results from Sub-task 2. "
        "Check for completeness of the enumeration, correctness of counts, and absence of overlaps or double counting. "
        "Challenge any assumptions and confirm the accuracy of the counts for bases near the threshold (e.g., bases 15 to 20). "
        "Provide a verification report with justifications grounded in explicit enumeration data. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_1_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1_5 = self.max_round
    all_thinking_1_5 = [[] for _ in range(N_max_1_5)]
    all_answer_1_5 = [[] for _ in range(N_max_1_5)]
    subtask_desc_1_5 = {
        "subtask_id": "stage_1_5.subtask_3",
        "instruction": debate_instruction_1_5,
        "context": ["user query", thinking_1_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_5):
        for i, agent in enumerate(debate_agents_1_5):
            if r == 0:
                thinking_1_5, answer_1_5 = await agent([taskInfo, thinking_1_2], debate_instruction_1_5, r, is_sub_task=True)
            else:
                input_infos_1_5 = [taskInfo, thinking_1_2] + all_thinking_1_5[r-1]
                thinking_1_5, answer_1_5 = await agent(input_infos_1_5, debate_instruction_1_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying enumeration results, thinking: {thinking_1_5.content}; answer: {answer_1_5.content}")
            all_thinking_1_5[r].append(thinking_1_5)
            all_answer_1_5[r].append(answer_1_5)
    final_decision_agent_1_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_5, answer_1_5 = await final_decision_agent_1_5([taskInfo] + all_thinking_1_5[-1], "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a verification report.", is_sub_task=True)
    sub_tasks.append(f"Stage 1.5 Subtask 3 output: thinking - {thinking_1_5.content}; answer - {answer_1_5.content}")
    subtask_desc_1_5['response'] = {"thinking": thinking_1_5, "answer": answer_1_5}
    logs.append(subtask_desc_1_5)
    print("Step 1.5.3: ", sub_tasks[-1])

    # Stage 2: Identify minimal base with more than 10 b-eautiful integers
    reflect_instruction_2_1 = (
        "Sub-task 1: Identify the minimal base b ≥ 2 for which the count of b-eautiful integers exceeds 10 by analyzing the verified enumeration results from Stage 1.5. "
        "Justify the choice of minimal base with explicit reference to the enumeration and verification outputs. Avoid relying on assumptions or unverified data. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_1 = self.max_round
    cot_inputs_2_1 = [taskInfo, thinking_1_5, answer_1_5]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": reflect_instruction_2_1,
        "context": ["user query", thinking_1_5.content, answer_1_5.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, reflect_instruction_2_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, identifying minimal base, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    for i in range(N_max_2_1):
        feedback_2_1, correct_2_1 = await critic_agent_2_1([taskInfo, thinking_2_1], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_1.id}, providing feedback, thinking: {feedback_2_1.content}; answer: {correct_2_1.content}")
        if correct_2_1.content == "True":
            break
        cot_inputs_2_1.extend([thinking_2_1, feedback_2_1])
        thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, reflect_instruction_2_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, refining minimal base identification, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Stage 2 Subtask 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_1, answer_2_1, sub_tasks, agents)
    return final_answer, logs
