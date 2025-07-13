async def forward_26(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: For a fixed positive integer a, determine the number of finite nonempty sets B of positive integers "
        "such that max(B) = a. Derive a formula for this count and explain the reasoning, with context from the query."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing count of sets B with max(B)=a, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc_1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc_1)

    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the formula from Sub-task 1, aggregate the counts over all elements of A to form an equation "
        "relating the elements of A to the total number of sets (2024). Consider possible forms of A and the resulting equation."
    )
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, aggregating counts over A, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc_2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc_2)

    cot_sc_instruction_3 = (
        "Sub-task 3: Convert 2024 into its binary representation with explicit bit indexing starting from bit position 1 (least significant bit). "
        "List all set bits along with their positional values and verify that the sum of 2^(bit position - 1) equals 2024. "
        "Use a small panel of agents redundantly to confirm the binary representation and mapping from bit positions to values."
    )
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    subtask_desc_3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, converting and verifying binary of 2024, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    answer3_content = Counter(possible_answers_3).most_common(1)[0][0]
    thinking3 = thinkingmapping_3[answer3_content]
    answer3 = answermapping_3[answer3_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc_3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc_3)

    cot_sc_instruction_4a = (
        "Sub-task 4a: From the verified binary representation in Sub-task 3, deduce the candidate set A by mapping each set bit at position k to the element k in A (i.e., A = {k | bit k is set})."
    )
    cot_agents_4a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_4a = []
    thinkingmapping_4a = {}
    answermapping_4a = {}
    subtask_desc_4a = {
        "subtask_id": "subtask_4a",
        "instruction": cot_sc_instruction_4a,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking4a, answer4a = await cot_agents_4a[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4a[i].id}, deducing candidate set A from binary, thinking: {thinking4a.content}; answer: {answer4a.content}")
        possible_answers_4a.append(answer4a.content)
        thinkingmapping_4a[answer4a.content] = thinking4a
        answermapping_4a[answer4a.content] = answer4a
    answer4a_content = Counter(possible_answers_4a).most_common(1)[0][0]
    thinking4a = thinkingmapping_4a[answer4a_content]
    answer4a = answermapping_4a[answer4a_content]
    sub_tasks.append(f"Sub-task 4a output: thinking - {thinking4a.content}; answer - {answer4a.content}")
    subtask_desc_4a['response'] = {"thinking": thinking4a, "answer": answer4a}
    logs.append(subtask_desc_4a)

    cot_reflect_instruction_4b = (
        "Sub-task 4b: Perform a built-in consistency check by computing the sum of 2^(a-1) for all a in the candidate set A and compare it to 2024. "
        "If there is a mismatch, reject the candidate set A and trigger regeneration or refinement of the binary decomposition (Sub-task 3)."
    )
    cot_agent_4b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4b = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4b = self.max_round
    cot_inputs_4b = [taskInfo, thinking4a, answer4a]
    subtask_desc_4b = {
        "subtask_id": "subtask_4b",
        "instruction": cot_reflect_instruction_4b,
        "context": ["user query", "thinking of subtask 4a", "answer of subtask 4a"],
        "agent_collaboration": "Reflexion"
    }
    thinking4b, answer4b = await cot_agent_4b(cot_inputs_4b, cot_reflect_instruction_4b, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4b.id}, checking consistency of candidate A, thinking: {thinking4b.content}; answer: {answer4b.content}")
    for i in range(N_max_4b):
        feedback, correct = await critic_agent_4b([taskInfo, thinking4b, answer4b],
                                                "Please review the consistency check of sum of 2^(a-1) for candidate A against 2024.",
                                                i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4b.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4b.extend([thinking4b, answer4b, feedback])
        thinking4b, answer4b = await cot_agent_4b(cot_inputs_4b, cot_reflect_instruction_4b, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4b.id}, refining consistency check, thinking: {thinking4b.content}; answer: {answer4b.content}")
    sub_tasks.append(f"Sub-task 4b output: thinking - {thinking4b.content}; answer - {answer4b.content}")
    subtask_desc_4b['response'] = {"thinking": thinking4b, "answer": answer4b}
    logs.append(subtask_desc_4b)

    cot_sc_instruction_5 = (
        "Sub-task 5: Calculate the sum of the elements of A based on the candidate set A obtained and verified in Sub-task 4b. "
        "Explain the calculation clearly."
    )
    cot_agents_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_5 = []
    thinkingmapping_5 = {}
    answermapping_5 = {}
    subtask_desc_5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_sc_instruction_5,
        "context": ["user query", "thinking of subtask 4b", "answer of subtask 4b"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking5, answer5 = await cot_agents_5[i]([taskInfo, thinking4b, answer4b], cot_sc_instruction_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5[i].id}, calculating sum of elements of A, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers_5.append(answer5.content)
        thinkingmapping_5[answer5.content] = thinking5
        answermapping_5[answer5.content] = answer5
    answer5_content = Counter(possible_answers_5).most_common(1)[0][0]
    thinking5 = thinkingmapping_5[answer5_content]
    answer5 = answermapping_5[answer5_content]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc_5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc_5)

    debate_instruction_6 = (
        "Sub-task 6: Verify the final solution by explicitly calculating and comparing the sum of 2^(a-1) for the proposed set A against 2024. "
        "If any discrepancy is found, engage in a structured Debate or Reflexion among agents to argue over the inconsistency and collaboratively converge on a corrected set A. "
        "Implement an iterative feedback loop that, upon detecting numeric inconsistencies, automatically triggers a re-run or refinement of subtasks 3 or 4 before finalizing the answer. "
        "Return the final sum of elements of A alongside the verification result."
    )
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]
    subtask_desc_6 = {
        "subtask_id": "subtask_6",
        "instruction": debate_instruction_6,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "Debate"
    }

    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking5, answer5], debate_instruction_6, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying solution, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)

        if any("sum matches 2024" in ans.content.lower() or "correct" in ans.content.lower() for ans in all_answer6[r]):
            break
        else:
            cot_sc_instruction_3_retry = cot_sc_instruction_3
            cot_agents_3_retry = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
            possible_answers_3_retry = []
            thinkingmapping_3_retry = {}
            answermapping_3_retry = {}
            for i in range(N_sc):
                thinking3_retry, answer3_retry = await cot_agents_3_retry[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3_retry, is_sub_task=True)
                agents.append(f"CoT-SC retry agent {cot_agents_3_retry[i].id}, retrying binary conversion, thinking: {thinking3_retry.content}; answer: {answer3_retry.content}")
                possible_answers_3_retry.append(answer3_retry.content)
                thinkingmapping_3_retry[answer3_retry.content] = thinking3_retry
                answermapping_3_retry[answer3_retry.content] = answer3_retry
            answer3_content_retry = Counter(possible_answers_3_retry).most_common(1)[0][0]
            thinking3 = thinkingmapping_3_retry[answer3_content_retry]
            answer3 = answermapping_3_retry[answer3_content_retry]
            sub_tasks.append(f"Sub-task 3 retry output: thinking - {thinking3.content}; answer - {answer3.content}")
            subtask_desc_3['response'] = {"thinking": thinking3, "answer": answer3}
            logs.append(subtask_desc_3)

            cot_agents_4a_retry = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
            possible_answers_4a_retry = []
            thinkingmapping_4a_retry = {}
            answermapping_4a_retry = {}
            for i in range(N_sc):
                thinking4a_retry, answer4a_retry = await cot_agents_4a_retry[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4a, is_sub_task=True)
                agents.append(f"CoT-SC retry agent {cot_agents_4a_retry[i].id}, retrying candidate A deduction, thinking: {thinking4a_retry.content}; answer: {answer4a_retry.content}")
                possible_answers_4a_retry.append(answer4a_retry.content)
                thinkingmapping_4a_retry[answer4a_retry.content] = thinking4a_retry
                answermapping_4a_retry[answer4a_retry.content] = answer4a_retry
            answer4a_content = Counter(possible_answers_4a_retry).most_common(1)[0][0]
            thinking4a = thinkingmapping_4a_retry[answer4a_content]
            answer4a = answermapping_4a_retry[answer4a_content]
            sub_tasks.append(f"Sub-task 4a retry output: thinking - {thinking4a.content}; answer - {answer4a.content}")
            subtask_desc_4a['response'] = {"thinking": thinking4a, "answer": answer4a}
            logs.append(subtask_desc_4a)

            cot_inputs_4b = [taskInfo, thinking4a, answer4a]
            thinking4b, answer4b = await cot_agent_4b(cot_inputs_4b, cot_reflect_instruction_4b, 0, is_sub_task=True)
            agents.append(f"Reflexion CoT retry agent {cot_agent_4b.id}, retrying consistency check, thinking: {thinking4b.content}; answer: {answer4b.content}")
            for i in range(N_max_4b):
                feedback, correct = await critic_agent_4b([taskInfo, thinking4b, answer4b],
                                                        "Please review the consistency check of sum of 2^(a-1) for candidate A against 2024.",
                                                        i, is_sub_task=True)
                agents.append(f"Critic agent {critic_agent_4b.id}, retry feedback, thinking: {feedback.content}; answer: {correct.content}")
                if correct.content == "True":
                    break
                cot_inputs_4b.extend([thinking4b, answer4b, feedback])
                thinking4b, answer4b = await cot_agent_4b(cot_inputs_4b, cot_reflect_instruction_4b, i + 1, is_sub_task=True)
                agents.append(f"Reflexion CoT retry agent {cot_agent_4b.id}, refining consistency check, thinking: {thinking4b.content}; answer: {answer4b.content}")
            sub_tasks.append(f"Sub-task 4b retry output: thinking - {thinking4b.content}; answer - {answer4b.content}")
            subtask_desc_4b['response'] = {"thinking": thinking4b, "answer": answer4b}
            logs.append(subtask_desc_4b)

            thinking5, answer5 = await cot_agents_5[0]([taskInfo, thinking4b, answer4b], cot_sc_instruction_5, is_sub_task=True)
            agents.append(f"CoT-SC retry agent {cot_agents_5[0].id}, retrying sum calculation, thinking: {thinking5.content}; answer: {answer5.content}")
            sub_tasks.append(f"Sub-task 5 retry output: thinking - {thinking5.content}; answer - {answer5.content}")
            subtask_desc_5['response'] = {"thinking": thinking5, "answer": answer5}
            logs.append(subtask_desc_5)

    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1],
                                                    "Sub-task 6: Make final decision on verification and finalize the answer.",
                                                    is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing answer, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc_6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc_6)

    print("Step 1: ", sub_tasks[0])
    print("Step 2: ", sub_tasks[1])
    print("Step 3: ", sub_tasks[2])
    print("Step 4a: ", sub_tasks[3])
    print("Step 4b: ", sub_tasks[4])
    print("Step 5: ", sub_tasks[5])
    print("Step 6: ", sub_tasks[6])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs
