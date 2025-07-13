async def forward_5(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Organize given edge lengths
    cot_instruction_1 = (
        "Sub-task 1: Identify and organize the given edge lengths of tetrahedron ABCD precisely. "
        "Confirm the pairs of equal edges as AB = CD = sqrt(41), AC = BD = sqrt(80), and BC = AD = sqrt(89). "
        "Explicitly map each edge to the tetrahedron's vertices without assuming symmetry beyond the given equalities. "
        "Prepare these symbolic edge lengths for use in volume and area calculations, maintaining exact radical forms and avoiding decimal approximations. "
        "Avoid assigning coordinates or making unverified assumptions about the shape."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT | Reflexion"
    }
    thinking_1, answer_1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, organizing edges, thinking: {thinking_1.content}; answer: {answer_1.content}")
    sub_tasks.append(f"Sub-task stage_1.subtask_1 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {"thinking": thinking_1, "answer": answer_1}
    logs.append(subtask_desc_1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 2: Compute volume and face areas

    # Sub-task 1: Compute volume using Cayley-Menger determinant
    cot_instruction_2_1 = (
        "Sub-task 1: Compute the volume of tetrahedron ABCD using the exact symbolic edge lengths identified in stage_1.subtask_1. "
        "Apply the Cayley-Menger determinant formula with the six given squared edge lengths. "
        "Maintain symbolic radicals throughout and avoid decimal approximations or rounding. "
        "Verify the tetrahedron is non-degenerate by confirming the volume is positive and consistent with edge lengths."
    )
    cot_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", thinking_1.content, answer_1.content],
        "agent_collaboration": "CoT | Reflexion"
    }
    thinking_2_1, answer_2_1 = await cot_agent_2_1([taskInfo, thinking_1, answer_1], cot_instruction_2_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_1.id}, computing volume, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    sub_tasks.append(f"Sub-task stage_2.subtask_1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    # Sub-task 2: Calculate area of face ABC
    cot_instruction_2_2 = (
        "Sub-task 2: Calculate the area of face ABC using Heron's formula with exact edges AB, BC, and AC from stage_1.subtask_1. "
        "Compute the semi-perimeter symbolically and then the area expression without decimal rounding. "
        "Maintain radical expressions and simplify where possible. Document the exact symbolic area expression for face ABC."
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_instruction_2_2,
        "context": ["user query", thinking_1.content, answer_1.content],
        "agent_collaboration": "CoT | Reflexion"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2([taskInfo, thinking_1, answer_1], cot_instruction_2_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_2.id}, calculating area ABC, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task stage_2.subtask_2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    # Sub-task 3: Calculate area of face ABD
    cot_instruction_2_3 = (
        "Sub-task 3: Calculate the area of face ABD using Heron's formula with exact edges AB, BD, and AD from stage_1.subtask_1. "
        "Compute the semi-perimeter symbolically and then the area expression without decimal rounding. "
        "Maintain radical expressions and simplify where possible. Document the exact symbolic area expression for face ABD."
    )
    cot_agent_2_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_3 = {
        "subtask_id": "stage_2.subtask_3",
        "instruction": cot_instruction_2_3,
        "context": ["user query", thinking_1.content, answer_1.content],
        "agent_collaboration": "CoT | Reflexion"
    }
    thinking_2_3, answer_2_3 = await cot_agent_2_3([taskInfo, thinking_1, answer_1], cot_instruction_2_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_3.id}, calculating area ABD, thinking: {thinking_2_3.content}; answer: {answer_2_3.content}")
    sub_tasks.append(f"Sub-task stage_2.subtask_3 output: thinking - {thinking_2_3.content}; answer - {answer_2_3.content}")
    subtask_desc_2_3['response'] = {"thinking": thinking_2_3, "answer": answer_2_3}
    logs.append(subtask_desc_2_3)
    print("Step 2.3: ", sub_tasks[-1])

    # Sub-task 4: Calculate area of face ACD
    cot_instruction_2_4 = (
        "Sub-task 4: Calculate the area of face ACD using Heron's formula with exact edges AC, CD, and AD from stage_1.subtask_1. "
        "Compute the semi-perimeter symbolically and then the area expression without decimal rounding. "
        "Maintain radical expressions and simplify where possible. Document the exact symbolic area expression for face ACD."
    )
    cot_agent_2_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_4 = {
        "subtask_id": "stage_2.subtask_4",
        "instruction": cot_instruction_2_4,
        "context": ["user query", thinking_1.content, answer_1.content],
        "agent_collaboration": "CoT | Reflexion"
    }
    thinking_2_4, answer_2_4 = await cot_agent_2_4([taskInfo, thinking_1, answer_1], cot_instruction_2_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_4.id}, calculating area ACD, thinking: {thinking_2_4.content}; answer: {answer_2_4.content}")
    sub_tasks.append(f"Sub-task stage_2.subtask_4 output: thinking - {thinking_2_4.content}; answer - {answer_2_4.content}")
    subtask_desc_2_4['response'] = {"thinking": thinking_2_4, "answer": answer_2_4}
    logs.append(subtask_desc_2_4)
    print("Step 2.4: ", sub_tasks[-1])

    # Sub-task 5: Calculate area of face BCD
    cot_instruction_2_5 = (
        "Sub-task 5: Calculate the area of face BCD using Heron's formula with exact edges BC, CD, and BD from stage_1.subtask_1. "
        "Compute the semi-perimeter symbolically and then the area expression without decimal rounding. "
        "Maintain radical expressions and simplify where possible. Document the exact symbolic area expression for face BCD."
    )
    cot_agent_2_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_5 = {
        "subtask_id": "stage_2.subtask_5",
        "instruction": cot_instruction_2_5,
        "context": ["user query", thinking_1.content, answer_1.content],
        "agent_collaboration": "CoT | Reflexion"
    }
    thinking_2_5, answer_2_5 = await cot_agent_2_5([taskInfo, thinking_1, answer_1], cot_instruction_2_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_5.id}, calculating area BCD, thinking: {thinking_2_5.content}; answer: {answer_2_5.content}")
    sub_tasks.append(f"Sub-task stage_2.subtask_5 output: thinking - {thinking_2_5.content}; answer - {answer_2_5.content}")
    subtask_desc_2_5['response'] = {"thinking": thinking_2_5, "answer": answer_2_5}
    logs.append(subtask_desc_2_5)
    print("Step 2.5: ", sub_tasks[-1])

    # Sub-task 6: Verify and sum face areas
    reflect_instruction_2_6 = (
        "Sub-task 6: Verify the consistency and correctness of the four face areas calculated in subtasks 2, 3, 4, and 5. "
        "Confirm no assumptions of congruency were made without verification. "
        "Check that the sum of the four symbolic face areas is consistent with the tetrahedron's edge lengths and volume. "
        "Flag any anomalies or inconsistencies for re-computation. Return the total surface area S as an exact symbolic expression."
    )
    cot_agent_2_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_6 = {
        "subtask_id": "stage_2.subtask_6",
        "instruction": reflect_instruction_2_6,
        "context": [
            "user query",
            thinking_2_2.content, answer_2_2.content,
            thinking_2_3.content, answer_2_3.content,
            thinking_2_4.content, answer_2_4.content,
            thinking_2_5.content, answer_2_5.content
        ],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_6, answer_2_6 = await cot_agent_2_6([
        taskInfo,
        thinking_2_2, answer_2_2,
        thinking_2_3, answer_2_3,
        thinking_2_4, answer_2_4,
        thinking_2_5, answer_2_5
    ], reflect_instruction_2_6, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_6.id}, verifying face areas and summing surface area, thinking: {thinking_2_6.content}; answer: {answer_2_6.content}")
    sub_tasks.append(f"Sub-task stage_2.subtask_6 output: thinking - {thinking_2_6.content}; answer - {answer_2_6.content}")
    subtask_desc_2_6['response'] = {"thinking": thinking_2_6, "answer": answer_2_6}
    logs.append(subtask_desc_2_6)
    print("Step 2.6: ", sub_tasks[-1])

    # Stage 3: Compute inradius
    cot_sc_instruction_3_1 = (
        "Sub-task 1: Compute the inradius r of tetrahedron ABCD using the formula r = 3V / S, "
        "where V is the volume from stage_2.subtask_1 and S is the total surface area from stage_2.subtask_6. "
        "Use the exact symbolic expressions for volume and surface area, maintaining radical forms and avoiding decimal approximations. "
        "Verify that the tetrahedron is non-degenerate and the formula applies. Confirm the computed inradius is positive and consistent with the tetrahedron's geometry."
    )
    N_sc = self.max_sc
    cot_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_3_1 = []
    possible_thinkings_3_1 = []
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": cot_sc_instruction_3_1,
        "context": [
            "user query",
            thinking_2_1.content, answer_2_1.content,
            thinking_2_6.content, answer_2_6.content
        ],
        "agent_collaboration": "CoT | SC_CoT"
    }
    for i in range(N_sc):
        thinking_3_1, answer_3_1 = await cot_agents_3_1[i]([
            taskInfo, thinking_2_1, answer_2_1, thinking_2_6, answer_2_6
        ], cot_sc_instruction_3_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3_1[i].id}, computing inradius, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
        possible_answers_3_1.append(answer_3_1)
        possible_thinkings_3_1.append(thinking_3_1)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1([
        taskInfo
    ] + possible_answers_3_1 + possible_thinkings_3_1, "Sub-task 3.1: Synthesize and choose the most consistent answer for inradius calculation.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_3.subtask_1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 3.1: ", sub_tasks[-1])

    # Stage 4: Simplify inradius expression and compute final sum
    debate_instruction_4_1 = (
        "Sub-task 1: Simplify the exact symbolic expression of the inradius r obtained in stage_3.subtask_1 into the form (m√n)/p, "
        "where m and p are positive integers that are coprime, and n is a positive, square-free integer. "
        "Perform radical simplification, reduce fractions, and verify that the expression matches the required format exactly. "
        "After simplification, compute and return the sum m + n + p as the final answer. Include a verification step to ensure no errors occurred during simplification and that the inputs (volume and surface area) were consistent and exact. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_4_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4_1 = self.max_round
    all_thinking_4_1 = [[] for _ in range(N_max_4_1)]
    all_answer_4_1 = [[] for _ in range(N_max_4_1)]
    subtask_desc_4_1 = {
        "subtask_id": "stage_4.subtask_1",
        "instruction": debate_instruction_4_1,
        "context": ["user query", thinking_3_1.content, answer_3_1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4_1):
        for i, agent in enumerate(debate_agents_4_1):
            if r == 0:
                thinking_4_1, answer_4_1 = await agent([
                    taskInfo, thinking_3_1, answer_3_1
                ], debate_instruction_4_1, r, is_sub_task=True)
            else:
                input_infos_4_1 = [taskInfo, thinking_3_1, answer_3_1] + all_thinking_4_1[r-1] + all_answer_4_1[r-1]
                thinking_4_1, answer_4_1 = await agent(input_infos_4_1, debate_instruction_4_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, simplifying inradius, thinking: {thinking_4_1.content}; answer: {answer_4_1.content}")
            all_thinking_4_1[r].append(thinking_4_1)
            all_answer_4_1[r].append(answer_4_1)
    final_decision_agent_4_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_4_1, answer_4_1 = await final_decision_agent_4_1([
        taskInfo
    ] + all_thinking_4_1[-1] + all_answer_4_1[-1], "Sub-task 4.1: Final simplification and sum calculation.", is_sub_task=True)
    agents.append(f"Final Decision agent, calculating final sum m+n+p, thinking: {thinking_4_1.content}; answer: {answer_4_1.content}")
    sub_tasks.append(f"Sub-task stage_4.subtask_1 output: thinking - {thinking_4_1.content}; answer - {answer_4_1.content}")
    subtask_desc_4_1['response'] = {"thinking": thinking_4_1, "answer": answer_4_1}
    logs.append(subtask_desc_4_1)
    print("Step 4.1: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_4_1, answer_4_1, sub_tasks, agents)
    return final_answer, logs