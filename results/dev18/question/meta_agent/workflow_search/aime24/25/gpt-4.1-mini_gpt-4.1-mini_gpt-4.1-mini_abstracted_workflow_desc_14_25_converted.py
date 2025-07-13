async def forward_25(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Extract and verify given information
    cot_sc_instruction_0_1 = (
        "Sub-task 1: Identify and clearly state all given geometric elements and constraints: "
        "the convex equilateral hexagon ABCDEF, the parallelism of opposite sides (AB || DE, BC || EF, CD || FA), "
        "and the triangle formed by the extensions of sides AB, CD, and EF with side lengths 200, 240, and 300. "
        "Avoid mixing multiple objectives in one subtask."
    )
    cot_agents_0_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_0_1 = []
    possible_thinkings_0_1 = []
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_sc_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_agents_0_1[i]([taskInfo], cot_sc_instruction_0_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_1[i].id}, identifying geometric elements, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_1.append(answer)
        possible_thinkings_0_1.append(thinking)
    final_decision_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_1, answer_0_1 = await final_decision_agent_0_1([taskInfo] + possible_thinkings_0_1, "Sub-task 1: Synthesize and choose the most consistent identification of geometric elements.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_0.subtask_1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0.1: ", sub_tasks[-1])

    cot_sc_instruction_0_2 = (
        "Sub-task 2: Verify that the conditions of convexity, equilateral sides, and parallel opposite sides can coexist and define a valid hexagon configuration. "
        "Avoid assuming validity without explicit verification."
    )
    cot_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_agents_0_2[i]([taskInfo, thinking_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, verifying hexagon validity, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_2.append(answer)
        possible_thinkings_0_2.append(thinking)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo] + possible_thinkings_0_2, "Sub-task 2: Synthesize and choose the most consistent verification of hexagon validity.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_0.subtask_2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 0.2: ", sub_tasks[-1])

    cot_sc_instruction_0_3 = (
        "Sub-task 3: Clarify the geometric interpretation of the triangle formed by the extensions of AB, CD, and EF, including defining the intersection points of these extended lines and explaining how the triangle's side lengths correspond to distances between these points. "
        "Avoid assuming these correspondences without explicit definition."
    )
    cot_agents_0_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_0_3 = []
    possible_thinkings_0_3 = []
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_sc_instruction_0_3,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_agents_0_3[i]([taskInfo, thinking_0_1], cot_sc_instruction_0_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_3[i].id}, clarifying triangle interpretation, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_3.append(answer)
        possible_thinkings_0_3.append(thinking)
    final_decision_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_3, answer_0_3 = await final_decision_agent_0_3([taskInfo] + possible_thinkings_0_3, "Sub-task 3: Synthesize and choose the most consistent clarification of triangle formed by extended lines.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_0.subtask_3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 0.3: ", sub_tasks[-1])

    # Stage 1: Define coordinates and parametric lines using Debate pattern
    debate_instruction_1_1 = (
        "Sub-task 1: Establish a coordinate system and assign vertex A at the origin. Define vectors for sides AB, BC, and CD in terms of the hexagon side length s and two independent angles (e.g., theta and phi) to capture the shape, ensuring opposite sides are parallel. "
        "Avoid assuming vector sums or relations without explicit geometric justification. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    all_thinking_1_1 = [[] for _ in range(self.max_round)]
    all_answer_1_1 = [[] for _ in range(self.max_round)]
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": debate_instruction_1_1,
        "context": ["user query", thinking_0_2.content, thinking_0_3.content],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents_1_1):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_0_2, thinking_0_3], debate_instruction_1_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_0_2, thinking_0_3] + all_thinking_1_1[r-1]
                thinking, answer = await agent(input_infos, debate_instruction_1_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, defining coordinates, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_1_1[r].append(thinking)
            all_answer_1_1[r].append(answer)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo] + all_thinking_1_1[-1], "Sub-task 1: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_1.subtask_1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    debate_instruction_1_2 = (
        "Sub-task 2: Using the coordinates from subtask 1, derive explicit parametric equations for the lines containing sides AB, CD, and EF. "
        "Ensure these parametric forms correctly represent the extended lines beyond the hexagon edges. Avoid shortcuts or assumptions about line directions. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    all_thinking_1_2 = [[] for _ in range(self.max_round)]
    all_answer_1_2 = [[] for _ in range(self.max_round)]
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": debate_instruction_1_2,
        "context": ["user query", thinking_1_1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents_1_2):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_1_1], debate_instruction_1_2, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_1_1] + all_thinking_1_2[r-1]
                thinking, answer = await agent(input_infos, debate_instruction_1_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, deriving parametric lines, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_1_2[r].append(thinking)
            all_answer_1_2[r].append(answer)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo] + all_thinking_1_2[-1], "Sub-task 2: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_1.subtask_2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    debate_instruction_1_3 = (
        "Sub-task 3: Compute the exact coordinates of the intersection points P, Q, and R formed by the pairwise intersections of the extended lines AB & CD, CD & EF, and EF & AB, respectively. "
        "Avoid approximations or assumptions; solve symbolically in terms of s, theta, and phi. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    all_thinking_1_3 = [[] for _ in range(self.max_round)]
    all_answer_1_3 = [[] for _ in range(self.max_round)]
    subtask_desc_1_3 = {
        "subtask_id": "stage_1.subtask_3",
        "instruction": debate_instruction_1_3,
        "context": ["user query", thinking_1_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents_1_3):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_1_2], debate_instruction_1_3, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_1_2] + all_thinking_1_3[r-1]
                thinking, answer = await agent(input_infos, debate_instruction_1_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, computing intersections, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_1_3[r].append(thinking)
            all_answer_1_3[r].append(answer)
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await final_decision_agent_1_3([taskInfo] + all_thinking_1_3[-1], "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_1.subtask_3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 1.3: ", sub_tasks[-1])

    cot_sc_instruction_1_4 = (
        "Sub-task 4: Express the distances |PQ|, |QR|, and |RP| explicitly as functions of s, theta, and phi, corresponding to the triangle side lengths 200, 240, and 300. "
        "Avoid assuming these distances equal simple vector differences without derivation."
    )
    cot_agents_1_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_1_4 = []
    possible_thinkings_1_4 = []
    subtask_desc_1_4 = {
        "subtask_id": "stage_1.subtask_4",
        "instruction": cot_sc_instruction_1_4,
        "context": ["user query", thinking_1_3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_agents_1_4[i]([taskInfo, thinking_1_3], cot_sc_instruction_1_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_4[i].id}, expressing distances, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_1_4.append(answer)
        possible_thinkings_1_4.append(thinking)
    final_decision_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_4, answer_1_4 = await final_decision_agent_1_4([taskInfo] + possible_thinkings_1_4, "Sub-task 4: Synthesize and choose the most consistent expressions for distances.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_1.subtask_4 output: thinking - {thinking_1_4.content}; answer - {answer_1_4.content}")
    subtask_desc_1_4['response'] = {"thinking": thinking_1_4, "answer": answer_1_4}
    logs.append(subtask_desc_1_4)
    print("Step 1.4: ", sub_tasks[-1])

    # Stage 2: Formulate and solve system of equations
    debate_instruction_2_1 = (
        "Sub-task 1: Formulate the system of equations by setting |PQ|=200, |QR|=240, and |RP|=300 using the expressions derived in stage_1.subtask_4. "
        "Avoid relying on classical formulas without derivation; ensure all equations are consistent with the geometric model. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    all_thinking_2_1 = [[] for _ in range(self.max_round)]
    all_answer_2_1 = [[] for _ in range(self.max_round)]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": debate_instruction_2_1,
        "context": ["user query", thinking_1_4.content],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents_2_1):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_1_4], debate_instruction_2_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_1_4] + all_thinking_2_1[r-1]
                thinking, answer = await agent(input_infos, debate_instruction_2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, formulating system, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_2_1[r].append(thinking)
            all_answer_2_1[r].append(answer)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + all_thinking_2_1[-1], "Sub-task 1: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_2.subtask_1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    cot_sc_instruction_2_2 = (
        "Sub-task 2: Analyze and simplify the system of equations to eliminate theta and phi by leveraging the parallelism and convexity constraints of the hexagon. "
        "Avoid premature numeric substitution; aim for analytic or symbolic simplification where possible."
    )
    cot_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_2_2 = []
    possible_thinkings_2_2 = []
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_sc_instruction_2_2,
        "context": ["user query", thinking_2_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_agents_2_2[i]([taskInfo, thinking_2_1], cot_sc_instruction_2_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_2[i].id}, simplifying system, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_2_2.append(answer)
        possible_thinkings_2_2.append(thinking)
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_2, answer_2_2 = await final_decision_agent_2_2([taskInfo] + possible_thinkings_2_2, "Sub-task 2: Synthesize and choose the most consistent simplification of the system.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_2.subtask_2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2.2: ", sub_tasks[-1])

    debate_instruction_2_3 = (
        "Sub-task 3: Solve the simplified system numerically or analytically to find the hexagon side length s. "
        "Avoid accepting solutions without verifying geometric feasibility. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_2_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    all_thinking_2_3 = [[] for _ in range(self.max_round)]
    all_answer_2_3 = [[] for _ in range(self.max_round)]
    subtask_desc_2_3 = {
        "subtask_id": "stage_2.subtask_3",
        "instruction": debate_instruction_2_3,
        "context": ["user query", thinking_2_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents_2_3):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_2_2], debate_instruction_2_3, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_2_2] + all_thinking_2_3[r-1]
                thinking, answer = await agent(input_infos, debate_instruction_2_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, solving system, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_2_3[r].append(thinking)
            all_answer_2_3[r].append(answer)
    final_decision_agent_2_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_3, answer_2_3 = await final_decision_agent_2_3([taskInfo] + all_thinking_2_3[-1], "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_2.subtask_3 output: thinking - {thinking_2_3.content}; answer - {answer_2_3.content}")
    subtask_desc_2_3['response'] = {"thinking": thinking_2_3, "answer": answer_2_3}
    logs.append(subtask_desc_2_3)
    print("Step 2.3: ", sub_tasks[-1])

    # Stage 3: Verify and finalize solution
    debate_instruction_3_1 = (
        "Sub-task 1: Verify the convexity and orientation of the hexagon using the computed side length s and the corresponding angles theta and phi. "
        "Confirm that the hexagon is indeed convex and that opposite sides remain parallel. Avoid skipping these geometric sanity checks. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    all_thinking_3_1 = [[] for _ in range(self.max_round)]
    all_answer_3_1 = [[] for _ in range(self.max_round)]
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instruction_3_1,
        "context": ["user query", thinking_2_3.content, thinking_1_1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents_3_1):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_2_3, thinking_1_1], debate_instruction_3_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_2_3, thinking_1_1] + all_thinking_3_1[r-1]
                thinking, answer = await agent(input_infos, debate_instruction_3_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying convexity, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_3_1[r].append(thinking)
            all_answer_3_1[r].append(answer)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1([taskInfo] + all_thinking_3_1[-1], "Sub-task 1: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_3.subtask_1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step 3.1: ", sub_tasks[-1])

    debate_instruction_3_2 = (
        "Sub-task 2: Confirm that the computed side length s and the derived hexagon configuration satisfy all original problem constraints, including equilateral sides, parallel opposite sides, and the triangle side lengths formed by the extended lines. "
        "Avoid relying solely on numeric agreement; include geometric consistency checks. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_3_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    all_thinking_3_2 = [[] for _ in range(self.max_round)]
    all_answer_3_2 = [[] for _ in range(self.max_round)]
    subtask_desc_3_2 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": debate_instruction_3_2,
        "context": ["user query", thinking_3_1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents_3_2):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_3_1], debate_instruction_3_2, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_3_1] + all_thinking_3_2[r-1]
                thinking, answer = await agent(input_infos, debate_instruction_3_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, confirming constraints, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_3_2[r].append(thinking)
            all_answer_3_2[r].append(answer)
    final_decision_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_2, answer_3_2 = await final_decision_agent_3_2([taskInfo] + all_thinking_3_2[-1], "Sub-task 2: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_3.subtask_2 output: thinking - {thinking_3_2.content}; answer - {answer_3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking_3_2, "answer": answer_3_2}
    logs.append(subtask_desc_3_2)
    print("Step 3.2: ", sub_tasks[-1])

    reflect_inst_3_3 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_3_3 = (
        "Sub-task 3: Present the final simplified numeric value of the hexagon side length s with appropriate precision and clarity. "
        "Avoid presenting results without contextual verification. " + reflect_inst_3_3
    )
    cot_agent_3_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_3_3 = [taskInfo, thinking_3_2, thinking_3_1]
    subtask_desc_3_3 = {
        "subtask_id": "stage_3.subtask_3",
        "instruction": cot_reflect_instruction_3_3,
        "context": ["user query", thinking_3_2.content, thinking_3_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_3_3, answer_3_3 = await cot_agent_3_3(cot_inputs_3_3, cot_reflect_instruction_3_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3_3.id}, refining final solution, thinking: {thinking_3_3.content}; answer: {answer_3_3.content}")
    critic_inst_3_3 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(self.max_round):
        feedback, correct = await critic_agent_3_3([taskInfo, thinking_3_3], critic_inst_3_3, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3_3.extend([thinking_3_3, feedback])
        thinking_3_3, answer_3_3 = await cot_agent_3_3(cot_inputs_3_3, cot_reflect_instruction_3_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3_3.id}, refining final solution, thinking: {thinking_3_3.content}; answer: {answer_3_3.content}")
    sub_tasks.append(f"Sub-task stage_3.subtask_3 output: thinking - {thinking_3_3.content}; answer - {answer_3_3.content}")
    subtask_desc_3_3['response'] = {"thinking": thinking_3_3, "answer": answer_3_3}
    logs.append(subtask_desc_3_3)
    print("Step 3.3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_3, answer_3_3, sub_tasks, agents)
    return final_answer, logs
