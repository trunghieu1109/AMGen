async def forward_20(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Formalization and derivation

    # Subtask 1: Formal restatement of b-eautiful number definition (CoT)
    cot_instruction_0_1 = (
        "Sub-task 1: Formally restate the definition of a b-eautiful number, explicitly specifying the digit constraints "
        "for the two-digit number n = x*b + y in base b, ensuring 1 ≤ x ≤ b-1 and 0 ≤ y ≤ b-1, and clarify that n must be positive and have exactly two digits (no leading zero)."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_1 = {
        "subtask_id": "stage_0_subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, formalizing definition, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 0.1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0.1: ", sub_tasks[-1])

    # Subtask 2: Derive and validate key equation (x + y)^2 = x*b + y (CoT + Reflexion)
    cot_reflex_instruction_0_2 = (
        "Sub-task 2: Derive and validate the key equation (x + y)^2 = x*b + y from the problem statement, confirming that the sum of digits squared equals the number itself, "
        "and verify that this equation correctly encodes the b-eautiful condition under the digit constraints defined in subtask 1."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_2 = {
        "subtask_id": "stage_0_subtask_2",
        "instruction": cot_reflex_instruction_0_2,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "CoT | Reflexion"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1], cot_reflex_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT-Reflexion agent {cot_agent_0_2.id}, deriving key equation, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 0.2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0.2: ", sub_tasks[-1])

    # Subtask 3: Analyze implications of equation and derive formula for enumeration (CoT)
    cot_instruction_0_3 = (
        "Sub-task 3: Analyze the implications of the equation (x + y)^2 = x*b + y on the possible values of x, y, and b, including explicit bounds on the sum s = x + y, "
        "and derive the formula x = (s^2 - y) / b to prepare for systematic enumeration. Emphasize digit bounds and integer divisibility conditions to prune the search space."
    )
    cot_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_3 = {
        "subtask_id": "stage_0_subtask_3",
        "instruction": cot_instruction_0_3,
        "context": ["user query", thinking_0_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_3, answer_0_3 = await cot_agent_0_3([taskInfo, thinking_0_2], cot_instruction_0_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_3.id}, analyzing equation implications, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Sub-task 0.3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 0.3: ", sub_tasks[-1])

    # Stage 1: Enumeration and counting

    # Subtask 1: For fixed base b, enumerate all valid (x,y) pairs using formula and digit constraints (SC_CoT)
    cot_sc_instruction_1_1 = (
        "Sub-task 1: For a fixed base b, enumerate all possible sums s = x + y where s is a positive integer, and for each s, compute candidate digit pairs (x, y) satisfying (x + y)^2 = x*b + y, "
        "using the formula x = (s^2 - y) / b. Apply digit constraints 1 ≤ x ≤ b-1 and 0 ≤ y ≤ b-1 to filter valid pairs. Avoid brute-force enumeration over all (x, y) pairs by leveraging the analytical formula to prune invalid candidates. "
        "Record all valid (x, y) pairs and their corresponding n = x*b + y."
    )
    N_sc = self.max_sc
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    subtask_desc_1_1 = {
        "subtask_id": "stage_1_subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_3.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    for i in range(N_sc):
        thinking_1_1, answer_1_1 = await cot_agents_1_1[i]([taskInfo, thinking_0_3], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, enumerating valid digit pairs for base b, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
        possible_answers_1_1.append(answer_1_1)
        possible_thinkings_1_1.append(thinking_1_1)

    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + possible_thinkings_1_1, "Sub-task 1: Synthesize and choose the most consistent and correct enumeration method and results for valid (x,y) pairs per base b." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    # Subtask 2: Count number of b-eautiful numbers for each base b and record valid pairs (SC_CoT)
    cot_sc_instruction_1_2 = (
        "Sub-task 2: Count the number of b-eautiful numbers found for the given base b from the enumeration in subtask 1, and record this count along with the list of valid (x, y) pairs and their corresponding n values. Maintain detailed logs for debugging and verification."
    )
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    subtask_desc_1_2 = {
        "subtask_id": "stage_1_subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    for i in range(N_sc):
        thinking_1_2, answer_1_2 = await cot_agents_1_2[i]([taskInfo, thinking_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, counting b-eautiful numbers per base, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
        possible_answers_1_2.append(answer_1_2)
        possible_thinkings_1_2.append(thinking_1_2)

    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + possible_thinkings_1_2, "Sub-task 2: Synthesize and choose the most consistent and correct counts and records of b-eautiful numbers per base." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    # Subtask 3: Iterate over bases b from 2 upwards, apply enumeration and counting, stop when count > 10 (CoT + SC_CoT)
    cot_instruction_1_3 = (
        "Sub-task 3: Develop and implement a deterministic algorithm to iterate over increasing bases b starting from 2 upwards, applying the enumeration and counting subtasks for each base. Continue until a base b is found for which the count of b-eautiful numbers exceeds ten. Store the enumeration results and counts for all bases checked to support later verification."
    )
    cot_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_3 = {
        "subtask_id": "stage_1_subtask_3",
        "instruction": cot_instruction_1_3,
        "context": ["user query", thinking_1_2.content],
        "agent_collaboration": "CoT | SC_CoT"
    }
    thinking_1_3, answer_1_3 = await cot_agent_1_3([taskInfo, thinking_1_2], cot_instruction_1_3, is_sub_task=True)
    agents.append(f"CoT-SC agent {cot_agent_1_3.id}, iterating bases and applying enumeration and counting, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 1.3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 1.3: ", sub_tasks[-1])

    # Stage 2: Minimal base identification and verification

    # Subtask 1: Identify minimal base b ≥ 2 with count > 10 using enumeration data (Reflexion + SC_CoT)
    reflex_sc_instruction_2_1 = (
        "Sub-task 1: Identify the minimal base b ≥ 2 for which the count of b-eautiful numbers exceeds ten by analyzing the recorded counts from the enumeration process. Confirm minimality by checking counts for bases immediately below and above the candidate base. Provide a detailed justification referencing the enumeration data and counts."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_1 = self.max_round
    cot_inputs_2_1 = [taskInfo, thinking_1_3]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2_subtask_1",
        "instruction": reflex_sc_instruction_2_1,
        "context": ["user query", thinking_1_3.content],
        "agent_collaboration": "Reflexion | SC_CoT"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, reflex_sc_instruction_2_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, identifying minimal base, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    for i in range(N_max_2_1):
        feedback_2_1, correct_2_1 = await critic_agent_2_1([taskInfo, thinking_2_1], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_1.id}, providing feedback, thinking: {feedback_2_1.content}; answer: {correct_2_1.content}")
        if correct_2_1.content == "True":
            break
        cot_inputs_2_1.extend([thinking_2_1, feedback_2_1])
        thinking_2_1, answer_2_1 = await cot_agent_2_1(cot_inputs_2_1, reflex_sc_instruction_2_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_1.id}, refining minimal base identification, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    # Subtask 2: Verify correctness of identified minimal base by rechecking b-eautiful numbers and counts (Reflexion)
    reflex_instruction_2_2 = (
        "Sub-task 2: Verify the correctness of the identified minimal base b by rechecking the b-eautiful numbers for that base, listing all valid (x, y) pairs and their corresponding n values, and confirming that the count is indeed greater than ten. Perform sanity checks on digit bounds and the key equation for each pair to ensure no off-by-one or boundary errors."
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_2 = self.max_round
    cot_inputs_2_2 = [taskInfo, thinking_2_1]
    subtask_desc_2_2 = {
        "subtask_id": "stage_2_subtask_2",
        "instruction": reflex_instruction_2_2,
        "context": ["user query", thinking_2_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, reflex_instruction_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, verifying minimal base correctness, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    for i in range(N_max_2_2):
        feedback_2_2, correct_2_2 = await critic_agent_2_2([taskInfo, thinking_2_2], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_2.id}, providing feedback, thinking: {feedback_2_2.content}; answer: {correct_2_2.content}")
        if correct_2_2.content == "True":
            break
        cot_inputs_2_2.extend([thinking_2_2, feedback_2_2])
        thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, reflex_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refining verification, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_2, answer_2_2, sub_tasks, agents)
    return final_answer, logs
