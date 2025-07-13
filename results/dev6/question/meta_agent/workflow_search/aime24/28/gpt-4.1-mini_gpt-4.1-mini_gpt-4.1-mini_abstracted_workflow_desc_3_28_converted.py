async def forward_28(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Extract and clearly define all given geometric parameters and establish a precise coordinate system. "
        "Specifically, identify the torus parameters: major radius R=6, minor radius r=3, and the sphere radius S=11. "
        "Place the sphere center at the origin (0,0,0) and align the torus axis with the z-axis. "
        "Introduce a vertical shift parameter H representing the vertical displacement of the torus center relative to the sphere center. "
        "Write the torus surface equation incorporating H as (sqrt(x^2 + y^2) - 6)^2 + (z - H)^2 = 9. "
        "Clarify the meaning of the two tangent configurations (r_i and r_o) as corresponding to two distinct values of H where the torus is externally tangent to the sphere along circles on the torus surface. "
        "Avoid assuming ambiguous orientations or neglecting the vertical shift parameter. This setup is critical to correctly model the geometry and avoid previous errors."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    print(f"Starting Sub-task 1: {cot_instruction_1}")
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, extracting parameters and setting coordinate system, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_2a = (
        "Sub-task 2a: Derive the correct geometric conditions for external tangency between the torus and the sphere using the coordinate system and parameters defined in Sub-task 1. "
        "Formulate the tangency condition as a system of equations involving the vertical shift H and the z-coordinate(s) of the tangent circle(s) on the torus. "
        "Use a pure Chain-of-Thought (CoT) approach to reason through the problem, explicitly considering the distance between the sphere center and points on the torus surface, "
        "and the equality of radii sums at tangency. Avoid incorrect or oversimplified distance equations; instead, carefully incorporate the vertical shift H and the torus geometry. "
        "Provide a detailed derivation of the system of equations that must be solved to find H and the tangent circle parameters."
    )
    N_sc = self.max_sc
    cot_agents_2a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2a = []
    possible_thinkings_2a = []
    subtask_desc2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_instruction_2a,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Starting Sub-task 2a: {cot_instruction_2a}")
    for i in range(N_sc):
        thinking2a, answer2a = await cot_agents_2a[i]([taskInfo, thinking1, answer1], cot_instruction_2a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2a[i].id}, deriving tangency conditions, thinking: {thinking2a.content}; answer: {answer2a.content}")
        possible_answers_2a.append(answer2a)
        possible_thinkings_2a.append(thinking2a)

    final_decision_agent_2a = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_2a = "Sub-task 2a: Synthesize and choose the most consistent and correct geometric system of equations for tangency involving H and z."
    thinking2a, answer2a = await final_decision_agent_2a([taskInfo] + possible_answers_2a + possible_thinkings_2a, final_instr_2a, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking2a.content}; answer - {answer2a.content}")
    subtask_desc2a['response'] = {"thinking": thinking2a, "answer": answer2a}
    logs.append(subtask_desc2a)
    print("Step 2a: ", sub_tasks[-1])

    cot_instruction_2b = (
        "Sub-task 2b: Solve the system of equations derived in Sub-task 2a symbolically to find possible values of the vertical shift H and corresponding z-coordinates of the tangent circles on the torus. "
        "Use Chain-of-Thought reasoning combined with a small Self-Critique panel to verify algebraic correctness and consistency at each step. "
        "Perform algebraic manipulations carefully to avoid errors and ensure solutions are real and physically meaningful. "
        "After obtaining candidate solutions, numerically verify that they satisfy the original geometric constraints and represent valid external tangency configurations. "
        "Reject any extraneous or invalid solutions."
    )
    cot_agents_2b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_2b = []
    possible_thinkings_2b = []
    subtask_desc2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_instruction_2b,
        "context": ["user query", "thinking of subtask 2a", "answer of subtask 2a"],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Starting Sub-task 2b: {cot_instruction_2b}")
    for i in range(N_sc):
        thinking2b, answer2b = await cot_agents_2b[i]([taskInfo, thinking2a, answer2a], cot_instruction_2b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2b[i].id}, solving tangency system, thinking: {thinking2b.content}; answer: {answer2b.content}")
        possible_answers_2b.append(answer2b)
        possible_thinkings_2b.append(thinking2b)

    final_decision_agent_2b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_2b = "Sub-task 2b: Synthesize and choose the most consistent and correct solutions for H and z that satisfy tangency and geometric constraints."
    thinking2b, answer2b = await final_decision_agent_2b([taskInfo] + possible_answers_2b + possible_thinkings_2b, final_instr_2b, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking2b.content}; answer - {answer2b.content}")
    subtask_desc2b['response'] = {"thinking": thinking2b, "answer": answer2b}
    logs.append(subtask_desc2b)
    print("Step 2b: ", sub_tasks[-1])

    cot_instruction_3 = (
        "Sub-task 3: Using the validated values of H and tangent circle positions from Sub-task 2b, derive explicit formulas for the radii r_i and r_o of the tangent circles on the torus surface. "
        "Express these radii in simplest radical or fractional form, carefully relating them to the torus parameters and the vertical shift. "
        "Validate these formulas by checking their consistency with known geometric properties of the torus and sphere, such as the relationship between the major and minor radii and the sphere radius. "
        "Avoid algebraic errors and ensure the expressions are exact and verifiable."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", "thinking of subtask 2b", "answer of subtask 2b"],
        "agent_collaboration": "CoT"
    }
    print(f"Starting Sub-task 3: {cot_instruction_3}")
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2b, answer2b], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, deriving formulas for r_i and r_o, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_4 = (
        "Sub-task 4: Compute the numerical values of r_i and r_o from the formulas derived in Sub-task 3 using the given parameters. "
        "Confirm that both values are positive and correspond to valid tangent circles on the torus. "
        "Cross-verify these numerical results by substituting them back into the tangency conditions and the torus surface equation to confirm exact tangency. "
        "Maintain exactness by avoiding premature rounding; keep expressions in fractional or radical form where possible. "
        "This step ensures the numerical validity of the tangent circle radii."
    )
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_4 = []
    possible_thinkings_4 = []
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "SC_CoT"
    }
    print(f"Starting Sub-task 4: {cot_instruction_4}")
    for i in range(N_sc):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, computing numerical values of r_i and r_o, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4)
        possible_thinkings_4.append(thinking4)

    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_4 = "Sub-task 4: Synthesize and choose the most consistent and correct numerical values for r_i and r_o."
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + possible_answers_4 + possible_thinkings_4, final_instr_4, is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    debate_instruction_5 = (
        "Sub-task 5: Analyze the problem's requirement that the difference r_i - r_o be expressible as a rational fraction m/n with relatively prime positive integers m and n. "
        "Critically evaluate the computed difference from Sub-task 4. If the direct difference is irrational, explore alternative interpretations or related geometric quantities (such as difference of squares or other derived measures) that could yield a rational value consistent with the problem statement. "
        "Use Debate and Reflexion patterns to rigorously justify the choice of the quantity to express as m/n, avoiding premature acceptance of irrational values. "
        "Provide algebraic or geometric proof supporting the rationality of the final difference expression."
    )
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Debate"
    }
    print(f"Starting Sub-task 5: {debate_instruction_5}")
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing rationality and difference, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)

    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_5 = "Sub-task 5: Given all the above thinking and answers, reason carefully and provide the justified simplified fraction m/n for the difference or alternative quantity."
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], final_instr_5, is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    reflexion_instruction_6 = (
        "Sub-task 6: Simplify the rational difference r_i - r_o (or the justified alternative quantity from Sub-task 5) to lowest terms, ensuring m and n are relatively prime positive integers. "
        "Compute the sum m + n as required by the problem. Perform a final verification of the entire solution chain, including initial assumptions, derivations, simplifications, and the rationality condition. "
        "Reflect on potential errors or alternative interpretations and confirm the robustness and consistency of the final result. Return the final answer m + n alongside a detailed explanation of the verification process."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_6 = self.max_round
    cot_inputs_6 = [taskInfo, thinking5, answer5]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": reflexion_instruction_6,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "Reflexion"
    }
    print(f"Starting Sub-task 6: {reflexion_instruction_6}")
    thinking6, answer6 = await cot_agent_6(cot_inputs_6, reflexion_instruction_6, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_6.id}, simplifying and verifying final answer, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(N_max_6):
        feedback, correct = await critic_agent_6([taskInfo, thinking6, answer6], "Please review and provide limitations or confirm correctness. Output exactly 'True' if correct.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, feedback: {feedback.content}; correctness: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_6.extend([thinking6, answer6, feedback])
        thinking6, answer6 = await cot_agent_6(cot_inputs_6, reflexion_instruction_6, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_6.id}, refining final answer, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs
