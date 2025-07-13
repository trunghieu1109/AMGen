async def forward_25(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Formally represent the convex equilateral hexagon ABCDEF using vector notation or coordinate geometry. "
        "Define vectors for each side (AB, BC, CD, DE, EF, FA) with a chosen coordinate system that simplifies calculations (e.g., place vertex A at the origin and side AB along the positive x-axis). "
        "Explicitly incorporate the given conditions: the hexagon is convex, equilateral (all sides equal length s), and opposite sides are parallel (AB || DE, BC || EF, CD || FA). "
        "Introduce parameters (e.g., angles θ and φ) to represent directions of sides without assuming specific values prematurely. "
        "Derive and state the relationships between these vectors, including parallelism constraints expressed as vector equalities or scalar multiples, and equal magnitudes. "
        "Clearly state all assumptions and parameter definitions to avoid underspecification."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, representing hexagon vectors, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_2 = (
        "Sub-task 2: Identify the lines containing sides AB, CD, and EF using the vector or coordinate representations from Subtask 1. "
        "Explicitly derive the parametric equations of these lines. Compute the pairwise intersection points of these lines to form the triangle (label these points P, Q, and R). "
        "Derive explicit coordinate expressions for P, Q, and R in terms of the hexagon side length s and the parameters (e.g., θ, φ) introduced earlier. "
        "Express the side lengths of triangle PQR (i.e., lengths PQ, QR, and RP) symbolically in terms of s, θ, and φ. "
        "Document all algebraic steps and justify each manipulation without relying on classical results or proportionality arguments."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, representing triangle from extended sides, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    debate_instruction_3 = (
        "Sub-task 3: Rigorously verify the symbolic expressions derived in Subtask 2. "
        "Check the correctness and completeness of the coordinate derivations for points P, Q, and R, and the symbolic formulas for triangle side lengths PQ, QR, and RP. "
        "Identify and correct any algebraic or geometric inconsistencies. Confirm that the expressions are suitable for substitution of known triangle side lengths (200, 240, 300). "
        "This verification should be conducted by a Debate agent or multi-agent cross-checking to ensure no steps are skipped or assumptions unstated."
    )
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, temperature=0.5) for _ in self.debate_role]
    N_max_3 = self.max_round
    all_thinking3 = [[] for _ in range(N_max_3)]
    all_answer3 = [[] for _ in range(N_max_3)]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instruction_3,
        "context": ["user query", thinking1.content, answer1.content, thinking2.content, answer2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2, answer2], debate_instruction_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3, answer3 = await agent(input_infos_3, debate_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying symbolic expressions, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo] + all_thinking3[-1] + all_answer3[-1], "Sub-task 3: Synthesize and confirm verified symbolic expressions for triangle side lengths.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing verified symbolic expressions, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_4 = (
        "Sub-task 4: Using the verified symbolic expressions from Subtask 3, set up the system of algebraic equations relating the hexagon side length s and parameters θ, φ to the known triangle side lengths 200, 240, and 300. "
        "Solve this system step-by-step, showing all algebraic manipulations, substitutions, and justifications. "
        "Employ appropriate methods such as substitution, elimination, or numerical approximation as needed. "
        "After obtaining candidate solutions for s, θ, and φ, verify that these satisfy the geometric constraints of the hexagon: convexity, equilateral sides, and parallel opposite sides. "
        "Explicitly check angle ranges and vector directions to confirm convexity and parallelism. Document the solution process thoroughly to avoid unsubstantiated conclusions."
    )
    N_sc_4 = self.max_sc
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_4)]
    possible_answers_4 = []
    possible_thinkings_4 = []
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", thinking1.content, answer1.content, thinking2.content, answer2.content, thinking3.content, answer3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_4):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, solving system for hexagon side length, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4)
        possible_thinkings_4.append(thinking4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + possible_answers_4 + possible_thinkings_4, "Sub-task 4: Synthesize and choose the most consistent and correct solution for the hexagon side length.", is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing solution for hexagon side length, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    reflexion_instruction_5 = (
        "Sub-task 5: Critically verify the solution obtained in Subtask 4. "
        "Re-examine all derivations and assumptions, test alternative parameter values if applicable, and confirm that the computed hexagon side length s is consistent with all problem conditions. "
        "Perform numeric checks by substituting s, θ, and φ back into the expressions for triangle side lengths and hexagon side vectors. "
        "Identify any contradictions or unstated assumptions. Provide a final, justified answer for the hexagon side length. "
        "If inconsistencies arise, specify which subtasks require refinement and suggest corrective actions. "
        "This subtask should involve Reflexion and Debate collaboration to ensure thorough critical evaluation and consensus on the final solution."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_5 = self.max_round
    cot_inputs_5 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": reflexion_instruction_5,
        "context": ["user query", thinking1.content, answer1.content, thinking2.content, answer2.content, thinking3.content, answer3.content, thinking4.content, answer4.content],
        "agent_collaboration": "Reflexion | Debate"
    }
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, reflexion_instruction_5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, verifying and refining hexagon side length, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max_5):
        feedback, correct = await critic_agent_5([taskInfo, thinking5, answer5], "Please review the answer above and criticize where it might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, reflexion_instruction_5, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining solution, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
