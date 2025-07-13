async def forward_23(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Problem Domain and Constraints Definition using Chain-of-Thought
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)

    # Subtask 1: Identify and clearly state the domain of the problem
    cot_instruction_0_1 = (
        "Sub-task 1: Identify and clearly state the domain of the problem: all possible assignments of digits (0-9) "
        "to each of the six cells in a 2x3 grid, explicitly allowing leading zeros and digit repetition. Emphasize the standard digit constraints and clarify that each cell holds a single digit."
    )
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, identifying problem domain, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 1: ", sub_tasks[-1])

    # Subtask 2: Describe precisely how the numbers are formed from the grid
    cot_instruction_0_2 = (
        "Sub-task 2: Describe precisely how the numbers are formed from the grid: two 3-digit numbers formed by reading each row left to right, "
        "and three 2-digit numbers formed by reading each column top to bottom. Clarify the concatenation order and confirm allowance of leading zeros, using the example grid as reference."
    )
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, describing number formation, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 2: ", sub_tasks[-1])

    # Subtask 3: Explicitly state the sum constraints
    cot_instruction_0_3 = (
        "Sub-task 3: Explicitly state the sum constraints: the sum of the two row-formed 3-digit numbers equals 999, "
        "and the sum of the three column-formed 2-digit numbers equals 99. Emphasize these as the key conditions for valid digit assignments."
    )
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_instruction_0_3,
        "context": ["user query", thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_3, answer_0_3 = await cot_agent_0_3([taskInfo, thinking_0_2, answer_0_2], cot_instruction_0_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_3.id}, stating sum constraints, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 1: Algebraic Modeling and Validation with Self-Consistency and Reflexion
    cot_sc_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    reflexion_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)

    # Subtask 1: Formally represent digits and express sum constraints algebraically
    cot_sc_instruction_1_1 = (
        "Sub-task 1: Formally represent the digits in the grid as variables (a,b,c for first row; d,e,f for second row) "
        "and express the two sum constraints as algebraic equations, explicitly incorporating place values for both rows and columns. "
        "Model the column sums as (10 * top_digit + bottom_digit). Avoid simplifying assumptions about digit sums."
    )
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_3.content, answer_0_3.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    for i in range(self.max_sc):
        thinking_1_1, answer_1_1 = await cot_sc_agents_1_1[i]([taskInfo, thinking_0_3, answer_0_3], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_1[i].id}, algebraic modeling, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
        possible_answers_1_1.append(answer_1_1)
        possible_thinkings_1_1.append(thinking_1_1)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + possible_answers_1_1 + possible_thinkings_1_1, "Sub-task 1: Synthesize and choose the most consistent algebraic model for the problem.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 4: ", sub_tasks[-1])

    # Subtask 2: Validate algebraic model against example grid
    cot_instruction_1_2 = (
        "Sub-task 2: Validate the algebraic model by testing the derived equations and assumptions against the provided example grid. "
        "Identify any contradictions or inconsistencies early to ensure the model accurately reflects the problem constraints."
    )
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_instruction_1_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2([taskInfo, thinking_1_1, answer_1_1], cot_instruction_1_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_2.id}, validating algebraic model, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 5: ", sub_tasks[-1])

    # Subtask 3: Reflexion to critically examine algebraic assumptions
    reflect_inst_1_3 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_1_3 = (
        "Sub-task 3: Critically reflect on the algebraic assumptions and constraints derived so far, especially regarding digit sums and carry-over effects. "
        "Challenge any oversimplifications and refine the model accordingly to ensure correctness before proceeding to enumeration. " + reflect_inst_1_3
    )
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": cot_reflect_instruction_1_3,
        "context": ["user query", thinking_1_1.content, answer_1_1.content, thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "Reflexion"
    }
    cot_inputs_1_3 = [taskInfo, thinking_1_1, answer_1_1, thinking_1_2, answer_1_2]
    thinking_1_3, answer_1_3 = await reflexion_agent_1_3(cot_inputs_1_3, cot_reflect_instruction_1_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {reflexion_agent_1_3.id}, reflecting on algebraic assumptions, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    for i in range(self.max_round):
        feedback_1_3, correct_1_3 = await critic_agent_1_3([taskInfo, thinking_1_3, answer_1_3], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_3.id}, providing feedback, thinking: {feedback_1_3.content}; answer: {correct_1_3.content}")
        if correct_1_3.content == "True":
            break
        cot_inputs_1_3.extend([thinking_1_3, answer_1_3, feedback_1_3])
        thinking_1_3, answer_1_3 = await reflexion_agent_1_3(cot_inputs_1_3, cot_reflect_instruction_1_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {reflexion_agent_1_3.id}, refining algebraic model, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 6: ", sub_tasks[-1])

    # Subtask 4: Analyze refined algebraic constraints to reduce search space
    cot_sc_agents_1_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    cot_sc_instruction_1_4 = (
        "Sub-task 4: Analyze the refined algebraic constraints to deduce relationships between digits and possible value ranges, "
        "explicitly considering carry-over in addition for both row and column sums. Use this analysis to reduce the search space for valid digit assignments."
    )
    subtask_desc_1_4 = {
        "subtask_id": "stage_1.subtask_4",
        "instruction": cot_sc_instruction_1_4,
        "context": ["user query", thinking_1_3.content, answer_1_3.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers_1_4 = []
    possible_thinkings_1_4 = []
    for i in range(self.max_sc):
        thinking_1_4, answer_1_4 = await cot_sc_agents_1_4[i]([taskInfo, thinking_1_3, answer_1_3], cot_sc_instruction_1_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_4[i].id}, analyzing refined constraints, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
        possible_answers_1_4.append(answer_1_4)
        possible_thinkings_1_4.append(thinking_1_4)
    final_decision_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_4, answer_1_4 = await final_decision_agent_1_4([taskInfo] + possible_answers_1_4 + possible_thinkings_1_4, "Sub-task 4: Synthesize and choose the most consistent analysis for digit relationships and ranges.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_1_4.content}; answer - {answer_1_4.content}")
    subtask_desc_1_4['response'] = {"thinking": thinking_1_4, "answer": answer_1_4}
    logs.append(subtask_desc_1_4)
    print("Step 7: ", sub_tasks[-1])

    # Subtask 5: Design systematic enumeration method respecting refined constraints
    cot_agent_1_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_instruction_1_5 = (
        "Sub-task 5: Design a systematic enumeration method or algorithm that respects the refined constraints, including place values and carry-over, "
        "to generate all valid digit assignments in the 2x3 grid. Ensure the method can handle the complexity introduced by the column sums and avoids invalid assumptions."
    )
    subtask_desc_1_5 = {
        "subtask_id": "stage_1.subtask_5",
        "instruction": cot_instruction_1_5,
        "context": ["user query", thinking_1_4.content, answer_1_4.content],
        "agent_collaboration": "CoT"
    }
    thinking_1_5, answer_1_5 = await cot_agent_1_5([taskInfo, thinking_1_4, answer_1_4], cot_instruction_1_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_5.id}, designing enumeration method, thinking: {thinking_1_5.content}; answer: {answer_1_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_1_5.content}; answer - {answer_1_5.content}")
    subtask_desc_1_5['response'] = {"thinking": thinking_1_5, "answer": answer_1_5}
    logs.append(subtask_desc_1_5)
    print("Step 8: ", sub_tasks[-1])

    # Stage 2: Execute Enumeration and Count Valid Assignments using Chain-of-Thought
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_instruction_2_1 = (
        "Sub-task 1: Execute the enumeration method to count the total number of valid digit assignments in the 2x3 grid "
        "that satisfy both sum constraints. Verify correctness by cross-checking with the example grid and ensure no duplicates or invalid solutions are included."
    )
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", thinking_1_5.content, answer_1_5.content],
        "agent_collaboration": "CoT"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1([taskInfo, thinking_1_5, answer_1_5], cot_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_1.id}, executing enumeration, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 9: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_1, answer_2_1, sub_tasks, agents)
    return final_answer, logs
