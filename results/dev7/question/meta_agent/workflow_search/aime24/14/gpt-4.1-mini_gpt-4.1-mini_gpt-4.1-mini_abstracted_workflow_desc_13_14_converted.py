async def forward_14(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = (
        "Sub-task 1: Derive parametric or algebraic representations for points A, B, C, and D on the hyperbola x^2/20 - y^2/24 = 1. "
        "Express these points using parameters that satisfy the hyperbola equation and the condition that the origin is the midpoint of diagonals AC and BD. "
        "Also, formulate the rhombus conditions: equal side lengths and perpendicular diagonals bisecting at the origin. "
        "Avoid assuming any orientation beyond these conditions."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, deriving parametric forms, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Sub-task 0 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)
    print("Step 0: ", sub_tasks[-1])

    reflexion_instruction_1 = (
        "Sub-task 1: Using the parametric forms from Stage 0, derive explicit formulas for BD^2 and AC^2 in terms of the parameters. "
        "Express the rhombus conditions: diagonals are perpendicular and bisect each other at the origin, and all sides are equal. "
        "Combine these with the hyperbola constraints to form algebraic relations suitable for optimization. "
        "Carefully reflect on the previous derivation to improve correctness and completeness."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_reflect_1 = self.max_round
    cot_inputs_1 = [taskInfo, thinking_0, answer_0]
    subtask_desc_1 = {
        "subtask_id": "subtask_1",
        "instruction": reflexion_instruction_1,
        "context": ["user query", "thinking of subtask 0", "answer of subtask 0"],
        "agent_collaboration": "Reflexion"
    }
    thinking_1, answer_1 = await cot_agent_1(cot_inputs_1, reflexion_instruction_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1.id}, deriving formulas and conditions, thinking: {thinking_1.content}; answer: {answer_1.content}")
    for i in range(N_reflect_1):
        feedback_1, correct_1 = await critic_agent_1([taskInfo, thinking_1, answer_1],
                                                    "Please review and provide limitations or errors in the derivation. If correct, output exactly 'True'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1.id}, feedback: {feedback_1.content}; correct: {correct_1.content}")
        if correct_1.content.strip() == "True":
            break
        cot_inputs_1.extend([thinking_1, answer_1, feedback_1])
        thinking_1, answer_1 = await cot_agent_1(cot_inputs_1, reflexion_instruction_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1.id}, refining derivation, thinking: {thinking_1.content}; answer: {answer_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {"thinking": thinking_1, "answer": answer_1}
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Analyze the algebraic relations from Stage 1 to reduce variables to a single parameter or minimal set. "
        "Use the perpendicularity and equal side length conditions to set up equations linking parameters. "
        "Formulate the optimization problem to find the supremum of BD^2 under these constraints. "
        "Consider domain restrictions from the hyperbola and rhombus geometry. "
        "Generate multiple independent reasoning attempts to ensure self-consistency."
    )
    N_sc_2 = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_2)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc_2 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_2):
        thinking_2, answer_2 = await cot_agents_2[i]([taskInfo, thinking_1, answer_1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyzing parameters and optimization, thinking: {thinking_2.content}; answer: {answer_2.content}")
        possible_answers_2.append(answer_2)
        possible_thinkings_2.append(thinking_2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2, answer_2 = await final_decision_agent_2([taskInfo] + possible_answers_2 + possible_thinkings_2,
                                                      "Sub-task 2: Synthesize and choose the most consistent and correct parameter reduction and optimization setup.",
                                                      is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing parameter analysis, thinking: {thinking_2.content}; answer: {answer_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {"thinking": thinking_2, "answer": answer_2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_3 = (
        "Sub-task 3: Select parameter values that maximize BD^2 while satisfying all constraints from previous stages. "
        "Verify the solution corresponds to a valid rhombus inscribed on the hyperbola with diagonals intersecting at the origin. "
        "Confirm the found value is the supremum of BD^2 for all such rhombi. "
        "Use multiple reasoning attempts for self-consistency and refine with Chain-of-Thought."
    )
    N_sc_3 = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_3)]
    possible_answers_3 = []
    possible_thinkings_3 = []
    subtask_desc_3 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_3):
        thinking_3, answer_3 = await cot_agents_3[i]([taskInfo, thinking_2, answer_2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, selecting and verifying max BD^2, thinking: {thinking_3.content}; answer: {answer_3.content}")
        possible_answers_3.append(answer_3)
        possible_thinkings_3.append(thinking_3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3, answer_3 = await final_decision_agent_3([taskInfo] + possible_answers_3 + possible_thinkings_3,
                                                      "Sub-task 3: Synthesize and verify the maximal BD^2 and its validity.",
                                                      is_sub_task=True)
    agents.append(f"Final Decision agent, verifying maximal BD^2, thinking: {thinking_3.content}; answer: {answer_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3['response'] = {"thinking": thinking_3, "answer": answer_3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3, answer_3, sub_tasks, agents)
    return final_answer, logs
