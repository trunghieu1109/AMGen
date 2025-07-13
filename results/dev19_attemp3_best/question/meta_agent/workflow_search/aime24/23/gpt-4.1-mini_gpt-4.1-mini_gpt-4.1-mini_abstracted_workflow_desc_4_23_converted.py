async def forward_23(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Derive and validate numeric representations and constraints

    # Subtask 1: Derive formal numeric representations and algebraic equations (SC_CoT)
    cot_sc_instruction_0_1 = (
        "Sub-task 1: Derive formal numeric representations of the two row numbers and three column numbers from the 2x3 grid digits, "
        "and translate the sum constraints (row sum = 999, column sum = 99) into algebraic equations involving the six digits."
    )
    N_sc = self.max_sc
    cot_sc_agents_0_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_0_1 = []
    possible_thinkings_0_1 = []
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_sc_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking, answer = await cot_sc_agents_0_1[i]([taskInfo], cot_sc_instruction_0_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0_1[i].id}, deriving numeric representations, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings_0_1.append(thinking)
        possible_answers_0_1.append(answer)
    final_decision_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_1, answer_0_1 = await final_decision_agent_0_1(
        [taskInfo] + possible_thinkings_0_1 + possible_answers_0_1,
        "Sub-task 1: Synthesize and choose the most consistent numeric representations and algebraic equations.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0.1: ", sub_tasks[-1])

    # Subtask 2: Validate derived numeric representations and constraints (CoT)
    cot_instruction_0_2 = (
        "Sub-task 2: Validate the derived numeric representations and constraints by checking the example grid and ensuring consistency with the problem statement, "
        "including allowance of leading zeros and digit repetition."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1, answer_0_1],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, validating numeric representations, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0.2: ", sub_tasks[-1])

    # Stage 1: Enumerate digit assignments satisfying row sum, then filter by column sum (Debate)

    # Subtask 1: Enumerate all digit assignments satisfying row sum = 999 (Debate)
    debate_instruction_1_1 = (
        "Sub-task 1: Enumerate all possible digit assignments (0-9) to the six grid cells that satisfy the row sum constraint (sum of two row numbers equals 999), "
        "using the formal representations from Stage 0. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_round_1_1 = self.max_round
    all_thinking_1_1 = [[] for _ in range(N_round_1_1)]
    all_answer_1_1 = [[] for _ in range(N_round_1_1)]
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": debate_instruction_1_1,
        "context": ["user query", thinking_0_2, answer_0_2],
        "agent_collaboration": "Debate"
    }
    for r in range(N_round_1_1):
        for i, agent in enumerate(debate_agents_1_1):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_0_2, answer_0_2], debate_instruction_1_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_0_2, answer_0_2] + all_thinking_1_1[r-1] + all_answer_1_1[r-1]
                thinking, answer = await agent(input_infos, debate_instruction_1_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, enumerating digit assignments for row sum, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_1_1[r].append(thinking)
            all_answer_1_1[r].append(answer)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1(
        [taskInfo, thinking_0_2, answer_0_2] + all_thinking_1_1[-1] + all_answer_1_1[-1],
        "Sub-task 1: Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    # Subtask 2: Filter digit assignments by column sum = 99 (Debate)
    debate_instruction_1_2 = (
        "Sub-task 2: From the digit assignments satisfying the row sum, select and verify those that also satisfy the column sum constraint (sum of three column numbers equals 99), "
        "ensuring all constraints are met simultaneously. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_round_1_2 = self.max_round
    all_thinking_1_2 = [[] for _ in range(N_round_1_2)]
    all_answer_1_2 = [[] for _ in range(N_round_1_2)]
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": debate_instruction_1_2,
        "context": ["user query", thinking_1_1, answer_1_1],
        "agent_collaboration": "Debate"
    }
    for r in range(N_round_1_2):
        for i, agent in enumerate(debate_agents_1_2):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_1_1, answer_1_1], debate_instruction_1_2, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_1_1, answer_1_1] + all_thinking_1_2[r-1] + all_answer_1_2[r-1]
                thinking, answer = await agent(input_infos, debate_instruction_1_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, filtering digit assignments by column sum, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_1_2[r].append(thinking)
            all_answer_1_2[r].append(answer)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2(
        [taskInfo, thinking_1_1, answer_1_1] + all_thinking_1_2[-1] + all_answer_1_2[-1],
        "Sub-task 2: Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    # Stage 2: Decompose sums into digit-level components and analyze carry-over (Reflexion)
    reflect_inst_2_1 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_2_1 = (
        "Sub-task 1: Decompose the numeric sums into digit-level components, analyze carry-over effects, and simplify the constraints to minimal forms to facilitate efficient verification and counting of valid digit assignments. "
        + reflect_inst_2_1
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_1 = self.max_round
    cot_inputs_2_1 = [taskInfo, thinking_1_2, answer_1_2]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_reflect_instruction_2_1,
        "context": ["user query", thinking_1_2, answer_1_2],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, analyzing carry-over and simplifying constraints, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    for i in range(N_max_2_1):
        critic_inst_2_1 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
        feedback, correct = await critic_agent_2_1([taskInfo, thinking_2_1, answer_2_1], critic_inst_2_1, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_1.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2_1.extend([thinking_2_1, answer_2_1, feedback])
        thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, refining analysis, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    # Stage 3: Aggregate and compute total number of valid assignments (SC_CoT)
    cot_sc_instruction_3_1 = (
        "Sub-task 1: Aggregate and combine the verified digit assignments from Stage 2 to compute the total number of valid ways to fill the grid satisfying both sum constraints."
    )
    cot_sc_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3_1 = []
    possible_thinkings_3_1 = []
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": cot_sc_instruction_3_1,
        "context": ["user query", thinking_2_1, answer_2_1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking, answer = await cot_sc_agents_3_1[i]([taskInfo, thinking_2_1, answer_2_1], cot_sc_instruction_3_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_3_1[i].id}, aggregating valid assignments, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings_3_1.append(thinking)
        possible_answers_3_1.append(answer)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1(
        [taskInfo, thinking_2_1, answer_2_1] + possible_thinkings_3_1 + possible_answers_3_1,
        "Sub-task 1: Synthesize and choose the most consistent and correct total count of valid assignments.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 3.1: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_1, answer_3_1, sub_tasks, agents)
    return final_answer, logs
