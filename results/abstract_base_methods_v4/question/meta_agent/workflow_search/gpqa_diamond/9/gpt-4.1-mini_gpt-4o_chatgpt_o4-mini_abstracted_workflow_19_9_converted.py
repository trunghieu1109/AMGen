async def forward_9(self, taskInfo):
    from collections import Counter
    import math
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1_1 = "Sub-task 1: Extract and clearly define the physical parameters (mass, radius, density, composition) for each of the four exoplanet choices from the query, ensuring correct identification of given and unknown values."
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1_1, answer1_1 = await cot_agent_1_1([taskInfo], cot_instruction_1_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_1.id}, extracting physical parameters, thinking: {thinking1_1.content}; answer: {answer1_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1_1.content}; answer - {answer1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking1_1, "answer": answer1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_1_2 = "Sub-task 2: Identify which planets have explicitly given densities and which require density calculation based on mass, radius, and composition assumptions, preparing for subsequent calculations."
    N = self.max_sc
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_1_2 = []
    thinkingmapping_1_2 = {}
    answermapping_1_2 = {}
    subtask_desc_1_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking1_2, answer1_2 = await cot_agents_1_2[i]([taskInfo, thinking1_1, answer1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, identifying density info, thinking: {thinking1_2.content}; answer: {answer1_2.content}")
        possible_answers_1_2.append(answer1_2.content)
        thinkingmapping_1_2[answer1_2.content] = thinking1_2
        answermapping_1_2[answer1_2.content] = answer1_2
    most_common_answer_1_2 = Counter(possible_answers_1_2).most_common(1)[0][0]
    thinking1_2 = thinkingmapping_1_2[most_common_answer_1_2]
    answer1_2 = answermapping_1_2[most_common_answer_1_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking1_2.content}; answer - {answer1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking1_2, "answer": answer1_2}
    logs.append(subtask_desc_1_2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_2_3 = (
        "Sub-task 3: Calculate the radius of the planet with the same composition as Earth but 5 times more massive (choice c) "
        "using the empirical mass-radius relation for Earth-like planets: R = R_earth × (M/M_earth)^0.27. Use R_earth = 1 Earth radius. "
        "Provide numeric calculation steps and results."
    )
    N = self.max_sc
    cot_agents_2_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2_3 = []
    thinkingmapping_2_3 = {}
    answermapping_2_3 = {}
    subtask_desc_2_3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_2_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2_3, answer2_3 = await cot_agents_2_3[i]([taskInfo, thinking1_2, answer1_2], cot_sc_instruction_2_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_3[i].id}, calculating radius for 5 Earth mass planet, thinking: {thinking2_3.content}; answer: {answer2_3.content}")
        possible_answers_2_3.append(answer2_3.content)
        thinkingmapping_2_3[answer2_3.content] = thinking2_3
        answermapping_2_3[answer2_3.content] = answer2_3
    most_common_answer_2_3 = Counter(possible_answers_2_3).most_common(1)[0][0]
    thinking2_3 = thinkingmapping_2_3[most_common_answer_2_3]
    answer2_3 = answermapping_2_3[most_common_answer_2_3]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking2_3.content}; answer - {answer2_3.content}")
    subtask_desc_2_3['response'] = {"thinking": thinking2_3, "answer": answer2_3}
    logs.append(subtask_desc_2_3)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_2_4 = (
        "Sub-task 4: Calculate the volume of the planet from subtask 3 using the formula for the volume of a sphere: V = (4/3)πR^3, "
        "ensuring units are consistent with Earth radius units. Provide numeric calculation steps and results."
    )
    cot_agent_2_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_2_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "CoT"
    }
    thinking2_4, answer2_4 = await cot_agent_2_4([taskInfo, thinking2_3, answer2_3], cot_instruction_2_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_4.id}, calculating volume for 5 Earth mass planet, thinking: {thinking2_4.content}; answer: {answer2_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking2_4.content}; answer - {answer2_4.content}")
    subtask_desc_2_4['response'] = {"thinking": thinking2_4, "answer": answer2_4}
    logs.append(subtask_desc_2_4)
    print("Step 4: ", sub_tasks[-1])

    cot_instruction_2_5 = (
        "Sub-task 5: Calculate the density of the planet from subtasks 3 and 4 by dividing its mass (5 Earth masses) by the calculated volume, "
        "converting to g/cm³ using Earth's density (5.51 g/cm³) as reference for unit consistency. Provide numeric calculation steps and results."
    )
    cot_agent_2_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_2_5,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "CoT"
    }
    thinking2_5, answer2_5 = await cot_agent_2_5([taskInfo, thinking2_3, answer2_3, thinking2_4, answer2_4], cot_instruction_2_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_5.id}, calculating density for 5 Earth mass planet, thinking: {thinking2_5.content}; answer: {answer2_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking2_5.content}; answer - {answer2_5.content}")
    subtask_desc_2_5['response'] = {"thinking": thinking2_5, "answer": answer2_5}
    logs.append(subtask_desc_2_5)
    print("Step 5: ", sub_tasks[-1])

    cot_sc_instruction_2_6 = (
        "Sub-task 6: Calculate the radius of the planet with the same composition as Earth but half the mass (choice d) using the empirical mass-radius relation: "
        "R = R_earth × (M/M_earth)^0.27. Provide numeric calculation steps and results."
    )
    N = self.max_sc
    cot_agents_2_6 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2_6 = []
    thinkingmapping_2_6 = {}
    answermapping_2_6 = {}
    subtask_desc_2_6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_sc_instruction_2_6,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2_6, answer2_6 = await cot_agents_2_6[i]([taskInfo, thinking1_2, answer1_2], cot_sc_instruction_2_6, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_6[i].id}, calculating radius for half Earth mass planet, thinking: {thinking2_6.content}; answer: {answer2_6.content}")
        possible_answers_2_6.append(answer2_6.content)
        thinkingmapping_2_6[answer2_6.content] = thinking2_6
        answermapping_2_6[answer2_6.content] = answer2_6
    most_common_answer_2_6 = Counter(possible_answers_2_6).most_common(1)[0][0]
    thinking2_6 = thinkingmapping_2_6[most_common_answer_2_6]
    answer2_6 = answermapping_2_6[most_common_answer_2_6]
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking2_6.content}; answer - {answer2_6.content}")
    subtask_desc_2_6['response'] = {"thinking": thinking2_6, "answer": answer2_6}
    logs.append(subtask_desc_2_6)
    print("Step 6: ", sub_tasks[-1])

    cot_instruction_2_7 = (
        "Sub-task 7: Calculate the volume of the planet from subtask 6 using the sphere volume formula V = (4/3)πR^3. Provide numeric calculation steps and results."
    )
    cot_agent_2_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_2_7,
        "context": ["user query", "thinking of subtask 6", "answer of subtask 6"],
        "agent_collaboration": "CoT"
    }
    thinking2_7, answer2_7 = await cot_agent_2_7([taskInfo, thinking2_6, answer2_6], cot_instruction_2_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_7.id}, calculating volume for half Earth mass planet, thinking: {thinking2_7.content}; answer: {answer2_7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking2_7.content}; answer - {answer2_7.content}")
    subtask_desc_2_7['response'] = {"thinking": thinking2_7, "answer": answer2_7}
    logs.append(subtask_desc_2_7)
    print("Step 7: ", sub_tasks[-1])

    cot_instruction_2_8 = (
        "Sub-task 8: Calculate the density of the planet from subtasks 6 and 7 by dividing its mass (0.5 Earth masses) by the calculated volume, "
        "converting to g/cm³ using Earth's density as reference. Provide numeric calculation steps and results."
    )
    cot_agent_2_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_instruction_2_8,
        "context": ["user query", "thinking of subtask 6", "answer of subtask 6", "thinking of subtask 7", "answer of subtask 7"],
        "agent_collaboration": "CoT"
    }
    thinking2_8, answer2_8 = await cot_agent_2_8([taskInfo, thinking2_6, answer2_6, thinking2_7, answer2_7], cot_instruction_2_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_8.id}, calculating density for half Earth mass planet, thinking: {thinking2_8.content}; answer: {answer2_8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking2_8.content}; answer - {answer2_8.content}")
    subtask_desc_2_8['response'] = {"thinking": thinking2_8, "answer": answer2_8}
    logs.append(subtask_desc_2_8)
    print("Step 8: ", sub_tasks[-1])

    cot_instruction_2_9 = (
        "Sub-task 9: Confirm the density of the planet with 2 Earth masses and density approximately 5.5 g/cm³ (choice b) as given, "
        "and the density of the Earth-mass and Earth-radius planet (choice a) as Earth's density (5.51 g/cm³). Provide explicit numeric confirmation."
    )
    cot_agent_2_9 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_9 = {
        "subtask_id": "subtask_9",
        "instruction": cot_instruction_2_9,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "CoT"
    }
    thinking2_9, answer2_9 = await cot_agent_2_9([taskInfo, thinking1_2, answer1_2], cot_instruction_2_9, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_9.id}, confirming given densities, thinking: {thinking2_9.content}; answer: {answer2_9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking2_9.content}; answer - {answer2_9.content}")
    subtask_desc_2_9['response'] = {"thinking": thinking2_9, "answer": answer2_9}
    logs.append(subtask_desc_2_9)
    print("Step 9: ", sub_tasks[-1])

    cot_instruction_3_10 = (
        "Sub-task 10: Aggregate the densities of all four planets (choices a, b, c, d) from subtasks 5, 8, 9, and given data, "
        "structuring the data in a comparable numeric format for further analysis."
    )
    cot_agent_3_10 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3_10 = {
        "subtask_id": "subtask_10",
        "instruction": cot_instruction_3_10,
        "context": ["user query", "thinking and answer of subtask 5", "thinking and answer of subtask 8", "thinking and answer of subtask 9"],
        "agent_collaboration": "CoT"
    }
    thinking3_10, answer3_10 = await cot_agent_3_10([taskInfo, thinking2_5, answer2_5, thinking2_8, answer2_8, thinking2_9, answer2_9], cot_instruction_3_10, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_10.id}, aggregating densities, thinking: {thinking3_10.content}; answer: {answer3_10.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking3_10.content}; answer - {answer3_10.content}")
    subtask_desc_3_10['response'] = {"thinking": thinking3_10, "answer": answer3_10}
    logs.append(subtask_desc_3_10)
    print("Step 10: ", sub_tasks[-1])

    cot_reflect_instruction_3_11 = (
        "Sub-task 11: Perform a self-consistency check and cross-validation of the calculated densities to identify and correct any arithmetic or logical errors, "
        "ensuring physical plausibility (e.g., higher mass Earth-like planets should have higher densities due to compression)."
    )
    cot_agent_3_11 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3_11 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs_3_11 = [taskInfo, thinking3_10, answer3_10]
    subtask_desc_3_11 = {
        "subtask_id": "subtask_11",
        "instruction": cot_reflect_instruction_3_11,
        "context": ["user query", "thinking and answer of subtask 10"],
        "agent_collaboration": "Reflexion"
    }
    thinking3_11, answer3_11 = await cot_agent_3_11(cot_inputs_3_11, cot_reflect_instruction_3_11, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3_11.id}, performing self-consistency check, thinking: {thinking3_11.content}; answer: {answer3_11.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent_3_11([taskInfo, thinking3_11, answer3_11], "Please review the density validation and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3_11.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip().lower() == "true":
            break
        cot_inputs_3_11.extend([thinking3_11, answer3_11, feedback])
        thinking3_11, answer3_11 = await cot_agent_3_11(cot_inputs_3_11, cot_reflect_instruction_3_11, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3_11.id}, refining validation, thinking: {thinking3_11.content}; answer: {answer3_11.content}")
    sub_tasks.append(f"Sub-task 11 output: thinking - {thinking3_11.content}; answer - {answer3_11.content}")
    subtask_desc_3_11['response'] = {"thinking": thinking3_11, "answer": answer3_11}
    logs.append(subtask_desc_3_11)
    print("Step 11: ", sub_tasks[-1])

    cot_instruction_3_12 = (
        "Sub-task 12: Compare the validated densities to determine which planet has the highest density, providing clear numeric comparison and reasoning."
    )
    cot_agent_3_12 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3_12 = {
        "subtask_id": "subtask_12",
        "instruction": cot_instruction_3_12,
        "context": ["user query", "thinking and answer of subtask 11"],
        "agent_collaboration": "CoT"
    }
    thinking3_12, answer3_12 = await cot_agent_3_12([taskInfo, thinking3_11, answer3_11], cot_instruction_3_12, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_12.id}, comparing densities, thinking: {thinking3_12.content}; answer: {answer3_12.content}")
    sub_tasks.append(f"Sub-task 12 output: thinking - {thinking3_12.content}; answer - {answer3_12.content}")
    subtask_desc_3_12['response'] = {"thinking": thinking3_12, "answer": answer3_12}
    logs.append(subtask_desc_3_12)
    print("Step 12: ", sub_tasks[-1])

    cot_instruction_3_13 = (
        "Sub-task 13: Map the planet with the highest density to its corresponding multiple-choice letter (a, b, c, or d) as requested by the query, "
        "ensuring the final answer is consistent with all prior calculations and validations."
    )
    cot_agent_3_13 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3_13 = {
        "subtask_id": "subtask_13",
        "instruction": cot_instruction_3_13,
        "context": ["user query", "thinking and answer of subtask 12"],
        "agent_collaboration": "CoT"
    }
    thinking3_13, answer3_13 = await cot_agent_3_13([taskInfo, thinking3_12, answer3_12], cot_instruction_3_13, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_13.id}, mapping highest density planet to choice letter, thinking: {thinking3_13.content}; answer: {answer3_13.content}")
    sub_tasks.append(f"Sub-task 13 output: thinking - {thinking3_13.content}; answer - {answer3_13.content}")
    subtask_desc_3_13['response'] = {"thinking": thinking3_13, "answer": answer3_13}
    logs.append(subtask_desc_3_13)
    print("Step 13: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3_13, answer3_13, sub_tasks, agents)
    return final_answer, logs
