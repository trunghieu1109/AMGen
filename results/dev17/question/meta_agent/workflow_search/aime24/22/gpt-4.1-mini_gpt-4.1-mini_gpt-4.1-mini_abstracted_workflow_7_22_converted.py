async def forward_22(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Formalize constraints

    # Subtask 1: Sum constraint (CoT)
    cot_instruction_0_1 = (
        "Sub-task 1: Formally represent the sum constraint: The list consists of positive integers whose sum equals 30. "
        "Emphasize that all elements are positive integers and the sum applies to the entire list."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, sum constraint, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0.1: ", sub_tasks[-1])

    # Subtask 2: Mode constraint (CoT)
    cot_instruction_0_2 = (
        "Sub-task 2: Formally represent the mode constraint: The unique mode of the list is 9, meaning 9 appears strictly more times than any other number and at least twice. "
        "Avoid assumptions about the exact frequency of 9 beyond these conditions."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, mode constraint, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0.2: ", sub_tasks[-1])

    # Subtask 3: Median constraint (CoT)
    cot_instruction_0_3 = (
        "Sub-task 3: Formally represent the median constraint: The median is a positive integer not present in the list, implying the list length is even. "
        "The median equals the average of the two middle elements, so the sum of these two elements equals twice the median integer. "
        "There is no restriction on the difference between the two middle elements. Emphasize that the median integer must not appear anywhere in the list."
    )
    cot_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_instruction_0_3,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_3, answer_0_3 = await cot_agent_0_3([taskInfo], cot_instruction_0_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_3.id}, median constraint, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Sub-task 0.3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 0.3: ", sub_tasks[-1])

    # Subtask 4: Analyze combined implications (SC_CoT)
    cot_sc_instruction_0_4 = (
        "Sub-task 4: Analyze the combined implications of the sum, mode, and median constraints on the list length, element frequencies, and possible values. "
        "Identify necessary conditions such as the list length being even, the minimum frequency of 9, and the nature of the median integer. "
        "Avoid premature assumptions or restrictions on list length or element distributions."
    )
    N_sc = self.max_sc
    cot_agents_0_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_0_4 = []
    possible_thinkings_0_4 = []
    subtask_desc_0_4 = {
        "subtask_id": "stage_0.subtask_4",
        "instruction": cot_sc_instruction_0_4,
        "context": ["user query", thinking_0_1, thinking_0_2, thinking_0_3],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_i, answer_i = await cot_agents_0_4[i]([taskInfo, thinking_0_1, thinking_0_2, thinking_0_3], cot_sc_instruction_0_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_4[i].id}, combined constraints analysis, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_0_4.append(answer_i)
        possible_thinkings_0_4.append(thinking_i)
    final_decision_agent_0_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_4, answer_0_4 = await final_decision_agent_0_4([taskInfo] + possible_thinkings_0_4, "Sub-task 4: Synthesize and choose the most consistent analysis of combined constraints.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 0.4 output: thinking - {thinking_0_4.content}; answer - {answer_0_4.content}")
    subtask_desc_0_4['response'] = {"thinking": thinking_0_4, "answer": answer_0_4}
    logs.append(subtask_desc_0_4)
    print("Step 0.4: ", sub_tasks[-1])

    # Stage 1: Candidate exploration and validation

    # Subtask 1: Enumerate plausible even list lengths (CoT)
    cot_instruction_1_1 = (
        "Sub-task 1: Enumerate all plausible even list lengths that could satisfy the median and sum constraints, starting from small lengths (4, 6) and extending to larger even lengths (8, 10) as needed. "
        "Provide reasoning for including or excluding certain lengths based on feasibility with sum=30 and mode=9 constraints."
    )
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query", thinking_0_4],
        "agent_collaboration": "CoT"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1([taskInfo, thinking_0_4], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, enumerate list lengths, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    # Subtask 2: Identify all possible median pairs for each candidate length (CoT)
    cot_instruction_1_2 = (
        "Sub-task 2: For each candidate list length, identify all possible pairs of middle elements whose sum is even and whose average (the median) is a positive integer not present in the list. "
        "Generate a comprehensive list of such median pairs without restricting the difference between the two middle elements. Document the median integer candidates for each list length."
    )
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query", thinking_1_1, thinking_0_3],
        "agent_collaboration": "CoT"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2([taskInfo, thinking_1_1, thinking_0_3], cot_instruction_1_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_2.id}, median pairs identification, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    # Subtask 3: Construct candidate lists for each median pair and length (SC_CoT)
    cot_sc_instruction_1_3 = (
        "Sub-task 3: For each candidate median pair and list length, systematically construct candidate lists of positive integers summing to 30, ensuring that 9 appears strictly more times than any other number, including cases where 9 appears three or more times. "
        "Use combinatorial or algorithmic search methods to generate all possible lists meeting the sum and median constraints before filtering by mode uniqueness. Avoid premature pruning of candidates."
    )
    N_sc_1_3 = self.max_sc
    cot_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc_1_3)]
    possible_answers_1_3 = []
    possible_thinkings_1_3 = []
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_sc_instruction_1_3,
        "context": ["user query", thinking_1_2, thinking_0_1, thinking_0_2],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_3):
        thinking_i, answer_i = await cot_agents_1_3[i]([taskInfo, thinking_1_2, thinking_0_1, thinking_0_2], cot_sc_instruction_1_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_3[i].id}, candidate list construction, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_3.append(answer_i)
        possible_thinkings_1_3.append(thinking_i)
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await final_decision_agent_1_3([taskInfo] + possible_thinkings_1_3, "Sub-task 3: Synthesize candidate lists meeting sum and median constraints with mode uniqueness enforcement.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 1.3: ", sub_tasks[-1])

    # Subtask 4: Verify mode uniqueness and median absence (SC_CoT)
    cot_sc_instruction_1_4 = (
        "Sub-task 4: Verify for each candidate list that the mode is uniquely 9 and that the median integer is not present in the list. "
        "If no valid list is found, provide explicit failure reasons (e.g., tied modes, median present in list) and pass this feedback back to candidate list construction to guide further attempts or refinements."
    )
    N_sc_1_4 = self.max_sc
    cot_agents_1_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc_1_4)]
    possible_answers_1_4 = []
    possible_thinkings_1_4 = []
    subtask_desc_1_4 = {
        "subtask_id": "stage_1.subtask_4",
        "instruction": cot_sc_instruction_1_4,
        "context": ["user query", thinking_1_3],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_4):
        thinking_i, answer_i = await cot_agents_1_4[i]([taskInfo, thinking_1_3], cot_sc_instruction_1_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_4[i].id}, mode uniqueness verification, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_answers_1_4.append(answer_i)
        possible_thinkings_1_4.append(thinking_i)
    final_decision_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_4, answer_1_4 = await final_decision_agent_1_4([taskInfo] + possible_thinkings_1_4, "Sub-task 4: Synthesize verification results and identify valid lists or failure reasons.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.4 output: thinking - {thinking_1_4.content}; answer - {answer_1_4.content}")
    subtask_desc_1_4['response'] = {"thinking": thinking_1_4, "answer": answer_1_4}
    logs.append(subtask_desc_1_4)
    print("Step 1.4: ", sub_tasks[-1])

    # Subtask 5: Iterative refinement and debate if no valid list found (Debate)
    debate_instruction_1_5 = (
        "Sub-task 5: If no valid list is found after initial attempts, iteratively refine assumptions and candidate generation by exploring longer even list lengths, adjusting frequency distributions of 9, or reconsidering median pairs. "
        "Incorporate multi-agent debate or critique to challenge previous assumptions and propose alternative constructions. Continue this iterative refinement until a valid list is found or exhaustive search is complete. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_1_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_1_5 = self.max_round
    all_thinking_1_5 = [[] for _ in range(N_max_1_5)]
    all_answer_1_5 = [[] for _ in range(N_max_1_5)]
    subtask_desc_1_5 = {
        "subtask_id": "stage_1.subtask_5",
        "instruction": debate_instruction_1_5,
        "context": ["user query", thinking_1_4],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_5):
        for i, agent in enumerate(debate_agents_1_5):
            if r == 0:
                thinking_1_5, answer_1_5 = await agent([taskInfo, thinking_1_4], debate_instruction_1_5, r, is_sub_task=True)
            else:
                input_infos_1_5 = [taskInfo, thinking_1_4] + all_thinking_1_5[r-1]
                thinking_1_5, answer_1_5 = await agent(input_infos_1_5, debate_instruction_1_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, iterative refinement, thinking: {thinking_1_5.content}; answer: {answer_1_5.content}")
            all_thinking_1_5[r].append(thinking_1_5)
            all_answer_1_5[r].append(answer_1_5)
    final_decision_agent_1_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_5, answer_1_5 = await final_decision_agent_1_5([taskInfo] + all_thinking_1_5[-1], "Sub-task 5: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, iterative refinement final decision, thinking: {thinking_1_5.content}; answer: {answer_1_5.content}")
    sub_tasks.append(f"Sub-task 1.5 output: thinking - {thinking_1_5.content}; answer - {answer_1_5.content}")
    subtask_desc_1_5['response'] = {"thinking": thinking_1_5, "answer": answer_1_5}
    logs.append(subtask_desc_1_5)
    print("Step 1.5: ", sub_tasks[-1])

    # Stage 2: Compute sum of squares for valid list(s)

    # Subtask 1: Compute sum of squares (Reflexion)
    reflect_inst_2_1 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_2_1 = (
        "Sub-task 1: For the valid list(s) identified, compute the sum of the squares of all items in the list. "
        "Clearly document the calculation steps and verify correctness. If multiple valid lists exist, confirm that the sum of squares is consistent or provide all possible results. "
        + reflect_inst_2_1
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_1 = self.max_round
    cot_inputs_2_1 = [taskInfo, thinking_1_5]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_reflect_instruction_2_1,
        "context": ["user query", thinking_1_5],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, sum of squares calculation, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    for i in range(N_max_2_1):
        feedback_2_1, correct_2_1 = await critic_agent_2_1([taskInfo, thinking_2_1], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_1.id}, feedback: {feedback_2_1.content}; correctness: {correct_2_1.content}")
        if correct_2_1.content == "True":
            break
        cot_inputs_2_1.extend([thinking_2_1, feedback_2_1])
        thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, cot_reflect_instruction_2_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, refining sum of squares calculation, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_1, answer_2_1, sub_tasks, agents)
    return final_answer, logs
