async def forward_20(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Formulate the mathematical conditions defining b-eautiful numbers precisely, "
        "including digit constraints (1 <= x <= b-1, 0 <= y <= b-1), the two-digit representation n = x*b + y, "
        "and the key equation x + y = sqrt(n). Explicitly clarify that sqrt(n) must be an integer and digits are integers, "
        "to avoid ambiguity and ensure correctness in subsequent reasoning."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, formulating conditions, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    debate_instruction_2 = (
        "Sub-task 2: Analyze the equation x + y = sqrt(x*b + y) to derive equivalent algebraic forms or inequalities "
        "that can help identify possible digit pairs (x,y) for a given base b. This includes squaring both sides and rearranging terms "
        "to express constraints on x, y, and b. The objective is to reduce the nonlinear problem to a more tractable form, avoiding brute force enumeration at this stage. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2 = self.max_round
    all_thinking2 = [[] for _ in range(N_max_2)]
    all_answer2 = [[] for _ in range(N_max_2)]
    subtask_desc2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": debate_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2):
        for i, agent in enumerate(debate_agents_2):
            if r == 0:
                thinking2, answer2 = await agent([taskInfo, thinking1, answer1], debate_instruction_2, r, is_sub_task=True)
            else:
                input_infos_2 = [taskInfo, thinking1, answer1] + all_thinking2[r-1] + all_answer2[r-1]
                thinking2, answer2 = await agent(input_infos_2, debate_instruction_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing algebraic forms, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking2[r].append(thinking2)
            all_answer2[r].append(answer2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking1, answer1] + all_thinking2[-1] + all_answer2[-1],
                                                     "Sub-task 2: Synthesize and choose the most consistent and correct algebraic constraints for the problem.",
                                                     is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_3_1 = (
        "Sub-task 1: Develop an efficient method to enumerate or characterize all two-digit numbers in base b that satisfy the b-eautiful condition for a fixed base b. "
        "Leverage the algebraic insights from stage_1 to prune impossible digit pairs and avoid unnecessary computations, addressing potential inefficiencies from naive enumeration."
    )
    N_sc = self.max_sc
    cot_sc_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    subtask_desc3_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_3_1,
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_3_1 = []
    possible_thinkings_3_1 = []
    for i in range(N_sc):
        thinking3_1, answer3_1 = await cot_sc_agents_3_1[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_3_1[i].id}, developing enumeration method, thinking: {thinking3_1.content}; answer: {answer3_1.content}")
        possible_answers_3_1.append(answer3_1)
        possible_thinkings_3_1.append(thinking3_1)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_1, answer3_1 = await final_decision_agent_3_1([taskInfo, thinking2, answer2] + possible_thinkings_3_1 + possible_answers_3_1,
                                                           "Sub-task 1: Synthesize and finalize the efficient enumeration method for b-eautiful numbers.",
                                                           is_sub_task=True)
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking3_1.content}; answer - {answer3_1.content}")
    subtask_desc3_1['response'] = {"thinking": thinking3_1, "answer": answer3_1}
    logs.append(subtask_desc3_1)
    print("Step 3.1: ", sub_tasks[-1])

    cot_instruction_3_2 = (
        "Sub-task 2: Implement or simulate the enumeration method for increasing values of b starting from 2, "
        "counting the number of b-eautiful numbers for each base. Carefully handle digit constraints and ensure correctness in counting, "
        "avoiding off-by-one or boundary errors. Identify the smallest base b for which the count exceeds ten."
    )
    cot_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_instruction_3_2,
        "context": ["user query", thinking3_1.content, answer3_1.content],
        "agent_collaboration": "CoT"
    }

    thinking3_2, answer3_2 = await cot_agent_3_2([taskInfo, thinking3_1, answer3_1], cot_instruction_3_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_2.id}, implementing enumeration and counting, thinking: {thinking3_2.content}; answer: {answer3_2.content}")
    sub_tasks.append(f"Sub-task 3.2 output: thinking - {thinking3_2.content}; answer - {answer3_2.content}")
    subtask_desc3_2['response'] = {"thinking": thinking3_2, "answer": answer3_2}
    logs.append(subtask_desc3_2)
    print("Step 3.2: ", sub_tasks[-1])

    reflect_inst_4_1 = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot_reflect_instruction_4_1 = (
        "Sub-task 1: Verify and validate the results obtained for the smallest base b with more than ten b-eautiful numbers. "
        "This includes cross-checking digit sums, square roots, and base representations to ensure no computational or logical errors occurred. "
        "The objective is to confirm the solution's correctness and robustness. " + reflect_inst_4_1
    )
    cot_agent_4_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4_1 = self.max_round
    cot_inputs_4_1 = [taskInfo, thinking3_2, answer3_2]
    subtask_desc4_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": cot_reflect_instruction_4_1,
        "context": ["user query", thinking3_2.content, answer3_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking4_1, answer4_1 = await cot_agent_4_1(cot_inputs_4_1, cot_reflect_instruction_4_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4_1.id}, verifying results, thinking: {thinking4_1.content}; answer: {answer4_1.content}")
    for i in range(N_max_4_1):
        feedback, correct = await critic_agent_4_1([taskInfo, thinking4_1, answer4_1],
                                                  "Please review and provide the limitations of provided solutions. "
                                                  "If you are absolutely sure it is correct, output exactly 'True' in 'correct'.",
                                                  i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4_1.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4_1.extend([thinking4_1, answer4_1, feedback])
        thinking4_1, answer4_1 = await cot_agent_4_1(cot_inputs_4_1, cot_reflect_instruction_4_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4_1.id}, refining verification, thinking: {thinking4_1.content}; answer: {answer4_1.content}")
    sub_tasks.append(f"Sub-task 4.1 output: thinking - {thinking4_1.content}; answer - {answer4_1.content}")
    subtask_desc4_1['response'] = {"thinking": thinking4_1, "answer": answer4_1}
    logs.append(subtask_desc4_1)
    print("Step 4.1: ", sub_tasks[-1])

    debate_instruction_4_2 = (
        "Sub-task 2: Provide a clear, concise explanation and justification of the final answer, including reasoning about why smaller bases do not meet the criterion and why the identified base does. "
        "This explanation should integrate insights from previous subtasks and address potential edge cases or exceptions. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_4_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4_2 = self.max_round
    all_thinking4_2 = [[] for _ in range(N_max_4_2)]
    all_answer4_2 = [[] for _ in range(N_max_4_2)]
    subtask_desc4_2 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": debate_instruction_4_2,
        "context": ["user query", thinking4_1.content, answer4_1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4_2):
        for i, agent in enumerate(debate_agents_4_2):
            if r == 0:
                thinking4_2, answer4_2 = await agent([taskInfo, thinking4_1, answer4_1], debate_instruction_4_2, r, is_sub_task=True)
            else:
                input_infos_4_2 = [taskInfo, thinking4_1, answer4_1] + all_thinking4_2[r-1] + all_answer4_2[r-1]
                thinking4_2, answer4_2 = await agent(input_infos_4_2, debate_instruction_4_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, explaining final answer, thinking: {thinking4_2.content}; answer: {answer4_2.content}")
            all_thinking4_2[r].append(thinking4_2)
            all_answer4_2[r].append(answer4_2)
    final_decision_agent_4_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4_2, answer4_2 = await final_decision_agent_4_2([taskInfo, thinking4_1, answer4_1] + all_thinking4_2[-1] + all_answer4_2[-1],
                                                             "Sub-task 4.2: Given all the above thinking and answers, reason over them carefully and provide a final answer.",
                                                             is_sub_task=True)
    sub_tasks.append(f"Sub-task 4.2 output: thinking - {thinking4_2.content}; answer - {answer4_2.content}")
    subtask_desc4_2['response'] = {"thinking": thinking4_2, "answer": answer4_2}
    logs.append(subtask_desc4_2)
    print("Step 4.2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4_2, answer4_2, sub_tasks, agents)
    return final_answer, logs
