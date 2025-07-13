async def forward_25(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_0 = (
        "Sub-task 1: Identify and verify all given elements and constraints of the problem. "
        "Confirm that ABCDEF is a convex equilateral hexagon with three pairs of opposite sides parallel. "
        "Enumerate the geometric implications, such as the hexagon being a parallelogon with translational symmetry. "
        "Clarify the meaning of the triangle formed by the extensions of sides AB, CD, and EF, explicitly stating that the triangle's side lengths (200, 240, 300) correspond to distances between intersection points of these extended lines. "
        "Avoid assumptions about angles or other dimensions not given, and explicitly note the need to assign which triangle side corresponds to which pair of extended lines in later steps."
    )
    N_sc_0 = self.max_sc
    cot_sc_agents_0 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_0)]
    possible_answers_0 = []
    possible_thinkings_0 = []
    subtask_desc_0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_sc_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_0):
        thinking0, answer0 = await cot_sc_agents_0[i]([taskInfo], cot_sc_instruction_0, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0[i].id}, verifying problem elements, thinking: {thinking0.content}; answer: {answer0.content}")
        possible_answers_0.append(answer0)
        possible_thinkings_0.append(thinking0)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent_0([taskInfo] + possible_answers_0 + possible_thinkings_0, "Sub-task 1: Synthesize and choose the most consistent verification of problem elements.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    agents.append(f"Final Decision agent, synthesizing verification, thinking: {thinking0.content}; answer: {answer0.content}")
    subtask_desc_0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc_0)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_1_1 = (
        "Sub-task 1: Set up a coordinate or vector system to represent the hexagon and its sides explicitly. "
        "Place point A at the origin (0,0). Define vectors for sides AB, BC, and CD using parameters: let the hexagon side length be s, and introduce an angle variable phi to represent the direction of side BC relative to AB. "
        "Express vectors for AB = a, BC = b, and CD = c = -(a + b) to satisfy the closed hexagon condition. "
        "Ensure opposite sides are parallel by construction. Derive explicit parametric equations for the lines containing sides AB, CD, and EF in terms of s and phi. "
        "Produce concrete algebraic formulas for the intersection points of these extended lines, avoiding vague or purely textual reasoning."
    )
    N_sc_1_1 = self.max_sc
    cot_sc_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_1)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking0, answer0],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_1):
        thinking1_1, answer1_1 = await cot_sc_agents_1_1[i]([taskInfo, thinking0, answer0], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_1[i].id}, deriving vector representations, thinking: {thinking1_1.content}; answer: {answer1_1.content}")
        possible_answers_1_1.append(answer1_1)
        possible_thinkings_1_1.append(thinking1_1)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_1, answer1_1 = await final_decision_agent_1_1([taskInfo] + possible_answers_1_1 + possible_thinkings_1_1, "Sub-task 1: Synthesize and choose the most consistent vector and coordinate representations.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1_1.content}; answer - {answer1_1.content}")
    agents.append(f"Final Decision agent, synthesizing vector setup, thinking: {thinking1_1.content}; answer: {answer1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking1_1, "answer": answer1_1}
    logs.append(subtask_desc_1_1)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_1_2 = (
        "Sub-task 2: Formulate explicit algebraic expressions for the distances between the intersection points of the extended lines AB, CD, and EF, which form the triangle with side lengths 200, 240, and 300. "
        "Assign each given triangle side length to a specific pair of extended lines (e.g., side length 200 corresponds to the distance between intersections of lines AB and CD, etc.) to remove ambiguity. "
        "Express these distances as functions of s and phi using the vector and coordinate formulas derived previously. "
        "Produce a system of two independent scalar equations relating s and phi to the known triangle side lengths, preparing for numeric solving. Avoid assumptions or skipping algebraic derivations."
    )
    N_sc_1_2 = self.max_sc
    cot_sc_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_2)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking0, answer0, thinking1_1, answer1_1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_2):
        thinking1_2, answer1_2 = await cot_sc_agents_1_2[i]([taskInfo, thinking0, answer0, thinking1_1, answer1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_2[i].id}, relating hexagon and triangle side lengths, thinking: {thinking1_2.content}; answer: {answer1_2.content}")
        possible_answers_1_2.append(answer1_2)
        possible_thinkings_1_2.append(thinking1_2)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_2, answer1_2 = await final_decision_agent_1_2([taskInfo] + possible_answers_1_2 + possible_thinkings_1_2, "Sub-task 2: Synthesize and choose the most consistent relationship between hexagon side length and triangle side lengths.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking1_2.content}; answer - {answer1_2.content}")
    agents.append(f"Final Decision agent, synthesizing side length relations, thinking: {thinking1_2.content}; answer: {answer1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking1_2, "answer": answer1_2}
    logs.append(subtask_desc_1_2)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_2_1 = (
        "Sub-task 1: Solve the system of equations derived in stage_1.subtask_2 for the unknowns s (the hexagon side length) and phi (the angle parameter). "
        "Use symbolic or numeric solving methods to find all possible solutions that satisfy the equations. "
        "Include checks for geometric validity, such as convexity constraints (e.g., 0 < phi < pi) and positivity of s. "
        "Document the solving process with intermediate numeric results and verify that solutions are consistent with the problem's geometric context."
    )
    N_sc_2_1 = self.max_sc
    cot_sc_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_2_1)]
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_instruction_2_1,
        "context": ["user query", thinking0, answer0, thinking1_1, answer1_1, thinking1_2, answer1_2],
        "agent_collaboration": "CoT"
    }
    for i in range(N_sc_2_1):
        thinking2_1, answer2_1 = await cot_sc_agents_2_1[i]([taskInfo, thinking0, answer0, thinking1_1, answer1_1, thinking1_2, answer1_2], cot_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2_1[i].id}, solving for hexagon side length, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
        possible_answers_2_1.append(answer2_1)
        possible_thinkings_2_1.append(thinking2_1)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2_1, answer2_1 = await final_decision_agent_2_1([taskInfo] + possible_answers_2_1 + possible_thinkings_2_1, "Sub-task 1: Synthesize and choose the most consistent hexagon side length solution.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking2_1.content}; answer - {answer2_1.content}")
    agents.append(f"Final Decision agent, synthesizing hexagon side length, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking2_1, "answer": answer2_1}
    logs.append(subtask_desc_2_1)
    print("Step 4: ", sub_tasks[-1])

    cot_instruction_2_2 = (
        "Sub-task 2: Perform numeric validation and sensitivity analysis on the solutions obtained. "
        "Substitute the candidate solutions for s and phi back into the third triangle side length equation (the one not used directly in the system) to verify consistency. "
        "Test alternative candidate solutions (e.g., s = 40 if it arises) and reject those that fail numeric checks or violate geometric constraints. "
        "Provide a reasoned justification for accepting the final solution(s)."
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_instruction_2_2,
        "context": ["user query", thinking2_1, answer2_1],
        "agent_collaboration": "CoT"
    }
    thinking2_2, answer2_2 = await cot_agent_2_2([taskInfo, thinking2_1, answer2_1], cot_instruction_2_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_2.id}, validating numeric solutions, thinking: {thinking2_2.content}; answer: {answer2_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_2.content}; answer - {answer2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking2_2, "answer": answer2_2}
    logs.append(subtask_desc_2_2)
    print("Step 5: ", sub_tasks[-1])

    debate_instruction_3_1 = (
        "Sub-task 1: Simplify the accepted hexagon side length solution to its minimal exact form or a suitable numerical approximation. "
        "Verify the solution by checking all original problem constraints, including the hexagon's equilateral and convex properties and the triangle side lengths formed by the extended lines. "
        "Document the verification process with explicit numeric checks and geometric reasoning to confirm correctness."
    )
    debate_instruction_3_2 = (
        "Sub-task 2: Synthesize the final answer with verification feedback. "
        "Confirm the uniqueness and correctness of the hexagon side length solution. "
        "Provide a clear, concise final answer along with a summary of the verification results and reasoning. "
        "Highlight any assumptions made and confirm that no unsupported claims remain. This subtask ensures the solution is fully justified and ready for presentation."
    )
    debate_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3_1 = self.max_round
    all_thinking_3_1 = [[] for _ in range(N_max_3_1)]
    all_answer_3_1 = [[] for _ in range(N_max_3_1)]
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instruction_3_1,
        "context": ["user query", thinking2_2, answer2_2],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3_1):
        for i, agent in enumerate(debate_agents_3_1):
            if r == 0:
                thinking3_1, answer3_1 = await agent([taskInfo, thinking2_2, answer2_2], debate_instruction_3_1, r, is_sub_task=True)
            else:
                input_infos_3_1 = [taskInfo, thinking2_2, answer2_2] + all_thinking_3_1[r-1] + all_answer_3_1[r-1]
                thinking3_1, answer3_1 = await agent(input_infos_3_1, debate_instruction_3_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, simplifying and verifying side length, thinking: {thinking3_1.content}; answer: {answer3_1.content}")
            all_thinking_3_1[r].append(thinking3_1)
            all_answer_3_1[r].append(answer3_1)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_1, answer3_1 = await final_decision_agent_3_1([taskInfo] + all_thinking_3_1[-1] + all_answer_3_1[-1], "Sub-task 1: Final decision on simplified and verified hexagon side length.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking3_1.content}; answer - {answer3_1.content}")
    agents.append(f"Final Decision agent, finalizing simplified side length, thinking: {thinking3_1.content}; answer: {answer3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking3_1, "answer": answer3_1}
    logs.append(subtask_desc_3_1)
    print("Step 6: ", sub_tasks[-1])

    reflect_inst_3_2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_3_2 = (
        "Sub-task 2: Synthesize the final answer with verification feedback. " + reflect_inst_3_2
    )
    cot_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3_2 = self.max_round
    cot_inputs_3_2 = [taskInfo, thinking0, answer0, thinking1_1, answer1_1, thinking1_2, answer1_2, thinking2_1, answer2_1, thinking2_2, answer2_2, thinking3_1, answer3_1]
    subtask_desc_3_2 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": cot_reflect_instruction_3_2,
        "context": ["user query", thinking0, answer0, thinking1_1, answer1_1, thinking1_2, answer1_2, thinking2_1, answer2_1, thinking2_2, answer2_2, thinking3_1, answer3_1],
        "agent_collaboration": "Reflexion"
    }
    thinking3_2, answer3_2 = await cot_agent_3_2(cot_inputs_3_2, cot_reflect_instruction_3_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3_2.id}, synthesizing final answer, thinking: {thinking3_2.content}; answer: {answer3_2.content}")
    for i in range(N_max_3_2):
        critic_inst_3_2 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'."
        feedback_3_2, correct_3_2 = await critic_agent_3_2([taskInfo, thinking3_2, answer3_2], "Please review and provide the limitations of provided solutions." + critic_inst_3_2, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3_2.id}, providing feedback, thinking: {feedback_3_2.content}; answer: {correct_3_2.content}")
        if correct_3_2.content == "True":
            break
        cot_inputs_3_2.extend([thinking3_2, answer3_2, feedback_3_2])
        thinking3_2, answer3_2 = await cot_agent_3_2(cot_inputs_3_2, cot_reflect_instruction_3_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3_2.id}, refining final answer, thinking: {thinking3_2.content}; answer: {answer3_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking3_2.content}; answer - {answer3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking3_2, "answer": answer3_2}
    logs.append(subtask_desc_3_2)
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3_2, answer3_2, sub_tasks, agents)
    return final_answer, logs
