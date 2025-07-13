async def forward_20(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Mathematical formulation and constraints (CoT)
    cot_instruction_1 = (
        "Sub-task 1: Formulate the mathematical definition of a b-eautiful number by expressing a two-digit number n in base b as n = x*b + y, "
        "with digit constraints 1 ≤ x ≤ b-1 and 0 ≤ y ≤ b-1, and establish the key equation x*b + y = (x + y)^2. "
        "Emphasize the necessity of integer digits and the relationship between the sum of digits and the square root of n."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, formulating definition, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_2 = (
        "Sub-task 2: Analyze the digit constraints and their implications on possible values of x, y, and b. "
        "Clarify that digits must be integers within their ranges, that the sum s = x + y is an integer equal to √n, "
        "and discuss the bounds on s given the base and digit constraints."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, analyzing digit constraints, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_instruction_3 = (
        "Sub-task 3: Derive inequalities and bounds on the sum of digits s and base b from the equation x*b + y = s^2, "
        "considering digit constraints and the fact that s = x + y. Identify feasible ranges for s and how they restrict possible digit pairs (x,y) for each base b."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", thinking2.content],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, deriving bounds, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 1: Enumeration and counting (CoT and SC_CoT)
    cot_instruction_4 = (
        "Sub-task 4: For a fixed base b, explicitly enumerate all possible digit pairs (x,y) with 1 ≤ x ≤ b-1 and 0 ≤ y ≤ b-1, "
        "and check which satisfy the equation x*b + y = (x + y)^2. Strictly enforce digit constraints and iteration bounds, "
        "and output a detailed list of all b-eautiful numbers (n) along with their digit pairs and sums."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", thinking3.content],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, enumerating digit pairs, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    cot_sc_instruction_5 = (
        "Sub-task 5: Develop or outline an efficient algorithm to count the number of b-eautiful numbers for any given base b using the enumeration results from subtask_4. "
        "Ensure the method avoids duplicates and correctly handles edge cases. The output should be the count of b-eautiful numbers for the given base."
    )
    N_sc = self.max_sc
    cot_agents_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_5 = []
    possible_thinkings_5 = []
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_sc_instruction_5,
        "context": ["user query", thinking4.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking5, answer5 = await cot_agents_5[i]([taskInfo, thinking4], cot_sc_instruction_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5[i].id}, counting b-eautiful numbers, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers_5.append(answer5)
        possible_thinkings_5.append(thinking5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + possible_thinkings_5, "Sub-task 5: Synthesize and choose the most consistent answer for counting b-eautiful numbers." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    # Stage 2: Enumeration for bases 2 to 20 and counting (CoT)
    cot_instruction_6a = (
        "Sub-task 6a: Systematically enumerate all b-eautiful numbers for each base b from 2 up to 20 by invoking the enumeration method from subtask_4. "
        "Collect and store the enumeration results for each base, ensuring strict digit constraints and no duplicates. "
        "Output a comprehensive table or list mapping each base b to its corresponding list of b-eautiful numbers."
    )
    cot_agent_6a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6a = {
        "subtask_id": "subtask_6a",
        "instruction": cot_instruction_6a,
        "context": ["user query", thinking4.content],
        "agent_collaboration": "CoT"
    }
    thinking6a, answer6a = await cot_agent_6a([taskInfo, thinking4], cot_instruction_6a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6a.id}, enumerating bases 2 to 20, thinking: {thinking6a.content}; answer: {answer6a.content}")
    sub_tasks.append(f"Sub-task 6a output: thinking - {thinking6a.content}; answer - {answer6a.content}")
    subtask_desc6a['response'] = {"thinking": thinking6a, "answer": answer6a}
    logs.append(subtask_desc6a)
    print("Step 6a: ", sub_tasks[-1])

    cot_sc_instruction_6b = (
        "Sub-task 6b: Analyze the enumeration data from subtask_6a to count the number of b-eautiful numbers for each base b. "
        "Identify the minimal base b for which the count exceeds ten. Output the minimal base along with the counts for all bases checked to support verification."
    )
    cot_agents_6b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_6b = []
    possible_thinkings_6b = []
    subtask_desc6b = {
        "subtask_id": "subtask_6b",
        "instruction": cot_sc_instruction_6b,
        "context": ["user query", thinking6a.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking6b, answer6b = await cot_agents_6b[i]([taskInfo, thinking6a], cot_sc_instruction_6b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6b[i].id}, analyzing counts, thinking: {thinking6b.content}; answer: {answer6b.content}")
        possible_answers_6b.append(answer6b)
        possible_thinkings_6b.append(thinking6b)
    final_decision_agent_6b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6b, answer6b = await final_decision_agent_6b([taskInfo] + possible_thinkings_6b, "Sub-task 6b: Synthesize and identify minimal base with more than ten b-eautiful numbers." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 6b output: thinking - {thinking6b.content}; answer - {answer6b.content}")
    subtask_desc6b['response'] = {"thinking": thinking6b, "answer": answer6b}
    logs.append(subtask_desc6b)
    print("Step 6b: ", sub_tasks[-1])

    # Stage 3: Verification of minimal base (Reflexion)
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_7 = (
        "Sub-task 7: Verify the correctness of the identified minimal base b by cross-checking the enumeration and counting results for all bases from 2 up to the minimal base. "
        "Confirm that no smaller base has more than ten b-eautiful numbers. Use the explicit enumeration data to rigorously validate counts and ensure no errors in digit constraints or counting. "
        "Employ interactive collaboration (e.g., Reflexion) to detect and resolve any discrepancies. "
        + reflect_inst
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_reflect = self.max_round
    cot_inputs_7 = [taskInfo, thinking6a, thinking6b]
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_reflect_instruction_7,
        "context": ["user query", thinking6a.content, thinking6b.content],
        "agent_collaboration": "Reflexion"
    }
    thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, verifying minimal base, thinking: {thinking7.content}; answer: {answer7.content}")
    critic_inst_7 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max_reflect):
        feedback7, correct7 = await critic_agent_7([taskInfo, thinking7], critic_inst_7, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7.id}, providing feedback, thinking: {feedback7.content}; answer: {correct7.content}")
        if correct7.content.strip() == "True":
            break
        cot_inputs_7.extend([thinking7, feedback7])
        thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_7.id}, refining verification, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs
