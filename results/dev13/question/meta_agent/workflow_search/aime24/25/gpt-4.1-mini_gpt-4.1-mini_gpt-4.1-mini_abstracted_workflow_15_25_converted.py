async def forward_25(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_N = self.max_sc

    # Stage 0: Setup and Parametric Definitions

    # Subtask 1: Vector representation of hexagon and parallelism conditions (SC_CoT)
    cot_sc_instruction_0_1 = (
        "Sub-task 1: Formally represent the convex equilateral hexagon ABCDEF with pairs of opposite sides parallel using vector notation. "
        "Define vectors for sides AB, BC, CD, DE, EF, and FA, and explicitly express the parallelism conditions (AB || DE, BC || EF, CD || FA) and the equilateral condition (all sides equal length s). "
        "Avoid introducing coordinate systems that contradict convexity or parallelism assumptions."
    )
    cot_sc_agents_0_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(cot_sc_N)]
    possible_answers_0_1 = []
    possible_thinkings_0_1 = []
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_sc_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(cot_sc_N):
        thinking, answer = await cot_sc_agents_0_1[i]([taskInfo], cot_sc_instruction_0_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0_1[i].id}, vector representation and parallelism, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_1.append(answer)
        possible_thinkings_0_1.append(thinking)
    final_decision_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_1, answer_0_1 = await final_decision_agent_0_1([taskInfo] + possible_answers_0_1 + possible_thinkings_0_1, "Sub-task 1: Synthesize vector representation and parallelism conditions for the hexagon.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    agents.append(f"Final Decision agent, vector representation and parallelism, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 1: ", sub_tasks[-1])

    # Subtask 2: Define lines AB, CD, EF and intersection points P, Q, R parametrically in terms of s and direction vectors (SC_CoT)
    cot_sc_instruction_0_2 = (
        "Sub-task 2: Using the vector representations from Sub-task 1, define the lines containing sides AB, CD, and EF. "
        "Formally define the intersection points P, Q, and R of these lines parametrically in terms of the hexagon side length s and the direction vectors of the sides. "
        "Express coordinates or vector forms of P, Q, and R explicitly without numeric assumptions."
    )
    cot_sc_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(cot_sc_N)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(cot_sc_N):
        thinking, answer = await cot_sc_agents_0_2[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0_2[i].id}, define lines and intersection points, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_2.append(answer)
        possible_thinkings_0_2.append(thinking)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo, thinking_0_1, answer_0_1] + possible_answers_0_2 + possible_thinkings_0_2, "Sub-task 2: Synthesize parametric definitions of intersection points P, Q, R.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    agents.append(f"Final Decision agent, parametric intersection points, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 2: ", sub_tasks[-1])

    # Subtask 3: Analyze and state geometric constraints and assumptions explicitly (CoT)
    cot_instruction_0_3 = (
        "Sub-task 3: Analyze and clearly state all geometric constraints and assumptions, including convexity, vertex ordering, and the nature of the triangle formed by the extended lines AB, CD, and EF. "
        "Explicitly clarify that the triangle is formed by the intersection points of these lines and that the hexagon is convex and equilateral with pairs of opposite sides parallel. "
        "Avoid ambiguous or contradictory assumptions."
    )
    cot_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_instruction_0_3,
        "context": ["user query", thinking_0_1.content, answer_0_1.content, thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "CoT"
    }
    thinking_0_3, answer_0_3 = await cot_agent_0_3([taskInfo, thinking_0_1, answer_0_1, thinking_0_2, answer_0_2], cot_instruction_0_3, is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    agents.append(f"CoT agent, geometric constraints and assumptions, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 1: Derive explicit formulas and solve system

    # Subtask 1: Derive explicit parametric formulas for |PQ|, |QR|, |RP| as functions of s and angles (SC_CoT)
    cot_sc_instruction_1_1 = (
        "Sub-task 1: Derive explicit parametric formulas for the side lengths |PQ|, |QR|, and |RP| of the triangle formed by points P, Q, and R as functions of the hexagon side length s and the three independent angles between the hexagon's side vectors. "
        "Use vector operations, trigonometric identities, and geometric properties to express these lengths concretely (e.g., |PQ| = s * f1(angles), etc.). "
        "Avoid placeholders or implicit functions; all expressions must be explicit and ready for numeric evaluation."
    )
    cot_sc_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(cot_sc_N)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", thinking_0_2.content, answer_0_2.content, thinking_0_3.content, answer_0_3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(cot_sc_N):
        thinking, answer = await cot_sc_agents_1_1[i]([taskInfo, thinking_0_2, answer_0_2, thinking_0_3, answer_0_3], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_1[i].id}, derive explicit formulas for triangle sides, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_1_1.append(answer)
        possible_thinkings_1_1.append(thinking)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo, thinking_0_2, answer_0_2, thinking_0_3, answer_0_3] + possible_answers_1_1 + possible_thinkings_1_1, "Sub-task 1: Synthesize explicit parametric formulas for |PQ|, |QR|, |RP|.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    agents.append(f"Final Decision agent, explicit formulas for triangle sides, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 4: ", sub_tasks[-1])

    # Subtask 2: Formulate system of equations 200 = s*f1, 240 = s*f2, 300 = s*f3 (SC_CoT)
    cot_sc_instruction_1_2 = (
        "Sub-task 2: Formulate the system of equations equating the derived triangle side lengths to the given lengths 200, 240, and 300: 200 = s * f1(angles), 240 = s * f2(angles), 300 = s * f3(angles). "
        "Identify the number of independent variables (s and two angles) and prepare the system for numeric or symbolic solving. "
        "Avoid assuming any values for s or angles without solving the system."
    )
    cot_sc_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(cot_sc_N)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(cot_sc_N):
        thinking, answer = await cot_sc_agents_1_2[i]([taskInfo, thinking_1_1, answer_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1_2[i].id}, formulate system of equations, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_1_2.append(answer)
        possible_thinkings_1_2.append(thinking)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo, thinking_1_1, answer_1_1] + possible_answers_1_2 + possible_thinkings_1_2, "Sub-task 2: Synthesize system of equations for numeric solving.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    agents.append(f"Final Decision agent, system of equations formulation, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 5: ", sub_tasks[-1])

    # Subtask 3: Solve nonlinear system numerically or symbolically (Debate)
    debate_instr_1_3 = (
        "Sub-task 3: Solve the nonlinear system of equations from Sub-task 2 numerically or symbolically to find the numeric values of the hexagon side length s and the two independent angles. "
        "Use appropriate numeric methods or algebraic manipulation. Avoid accepting unverified or assumed values; the solution must be derived from the system. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1_3 = self.max_round
    all_thinking_1_3 = [[] for _ in range(N_max_1_3)]
    all_answer_1_3 = [[] for _ in range(N_max_1_3)]
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": debate_instr_1_3,
        "context": ["user query", thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_3):
        for i, agent in enumerate(debate_agents_1_3):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_1_2, answer_1_2], debate_instr_1_3, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_1_2, answer_1_2] + all_thinking_1_3[r-1] + all_answer_1_3[r-1]
                thinking, answer = await agent(input_infos, debate_instr_1_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, solving nonlinear system, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_1_3[r].append(thinking)
            all_answer_1_3[r].append(answer)
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await final_decision_agent_1_3([taskInfo] + all_thinking_1_3[-1] + all_answer_1_3[-1], "Sub-task 3: Final decision on numeric solution for s and angles.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    agents.append(f"Final Decision agent, numeric solution, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 6: ", sub_tasks[-1])

    # Stage 2: Verification and Summary

    # Subtask 1: Verify computed s and angles satisfy all geometric constraints (Debate)
    debate_instr_2_1 = (
        "Sub-task 1: Verify that the computed hexagon side length s and angles satisfy all geometric constraints, including convexity, parallelism of opposite sides, and equilateral side conditions. "
        "Check that the solution is consistent with the problem statement and that the triangle side lengths match the given values within acceptable tolerance. "
        "Reject or revise any solution failing these checks. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_1 = self.max_round
    all_thinking_2_1 = [[] for _ in range(N_max_2_1)]
    all_answer_2_1 = [[] for _ in range(N_max_2_1)]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": debate_instr_2_1,
        "context": ["user query", thinking_1_3.content, answer_1_3.content, thinking_0_3.content, answer_0_3.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_1):
        for i, agent in enumerate(debate_agents_2_1):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_1_3, answer_1_3, thinking_0_3, answer_0_3], debate_instr_2_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_1_3, answer_1_3, thinking_0_3, answer_0_3] + all_thinking_2_1[r-1] + all_answer_2_1[r-1]
                thinking, answer = await agent(input_infos, debate_instr_2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verification of solution, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_2_1[r].append(thinking)
            all_answer_2_1[r].append(answer)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + all_thinking_2_1[-1] + all_answer_2_1[-1], "Sub-task 1: Final verification of computed hexagon side length and angles.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    agents.append(f"Final Decision agent, verification, thinking: {thinking_2_1.content}; answer: {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 7: ", sub_tasks[-1])

    # Subtask 2: Summarize and present final numeric value of hexagon side length s (Reflexion)
    reflect_inst_2_2 = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot_reflect_instruction_2_2 = (
        "Sub-task 2: Summarize and clearly present the final numeric value of the hexagon's side length s, including a concise explanation of how it was derived and verified. "
        "Provide a final conclusion consistent with all problem constraints and computations. Avoid introducing new assumptions or unverified claims. "
        + reflect_inst_2_2
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_2 = self.max_round
    cot_inputs_2_2 = [taskInfo, thinking_2_1, answer_2_1, thinking_1_3, answer_1_3]
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_reflect_instruction_2_2,
        "context": ["user query", thinking_2_1.content, answer_2_1.content, thinking_1_3.content, answer_1_3.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, summarizing final solution, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    for i in range(N_max_2_2):
        feedback, correct = await critic_agent_2_2([taskInfo, thinking_2_2, answer_2_2], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_2.id}, feedback round {i}, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_2_2.extend([thinking_2_2, answer_2_2, feedback])
        thinking_2_2, answer_2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refining summary round {i+1}, thinking: {thinking_2_2.content}; answer: {answer_2_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 8: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_2, answer_2_2, sub_tasks, agents)
    return final_answer, logs
