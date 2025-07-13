async def forward_21(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Geometric Data Preparation and Formalization

    # Subtask 1: Analyze geometric properties of the regular dodecagon (SC_CoT)
    cot_sc_instruction_0_1 = (
        "Sub-task 1: Analyze the geometric properties of the regular dodecagon, including precise vertex coordinates on the circumscribed circle, "
        "side lengths, and the full set of chords (all sides and diagonals). Compute chord directions and lengths relevant to rectangle formation. "
        "Avoid assuming rectangle vertices lie only at polygon vertices and prepare data for subsequent intersection analysis.")
    N_sc_0_1 = self.max_sc
    cot_agents_0_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_0_1)]
    possible_answers_0_1 = []
    possible_thinkings_0_1 = []
    subtask_desc_0_1 = {
        "subtask_id": "stage_0_subtask_1",
        "instruction": cot_sc_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_0_1):
        thinking, answer = await cot_agents_0_1[i]([taskInfo], cot_sc_instruction_0_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_1[i].id}, analyzing geometric properties, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_1.append(answer)
        possible_thinkings_0_1.append(thinking)
    final_decision_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_1, answer_0_1 = await final_decision_agent_0_1(
        [taskInfo] + possible_thinkings_0_1 + possible_answers_0_1,
        "Sub-task 1: Synthesize and choose the most consistent geometric analysis for the regular dodecagon.",
        is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 1: ", sub_tasks[-1])

    # Subtask 2: Construct complete line arrangement and compute all intersection points (Debate)
    debate_instruction_0_2 = (
        "Sub-task 2: Construct the complete line arrangement formed by all sides and diagonals of the dodecagon. "
        "Compute all intersection points of these lines inside the polygon, including intersections between diagonals and sides, and between diagonals themselves. "
        "Identify and record all line segments between intersection points that lie fully inside the polygon. "
        "This addresses the previous failure of ignoring partial chord segments and interior intersection points, enabling enumeration of rectangles formed by these segments. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.")
    debate_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_0_2 = self.max_round
    all_thinking_0_2 = [[] for _ in range(N_max_0_2)]
    all_answer_0_2 = [[] for _ in range(N_max_0_2)]
    subtask_desc_0_2 = {
        "subtask_id": "stage_0_subtask_2",
        "instruction": debate_instruction_0_2,
        "context": ["user query", thinking_0_1.content, answer_0_1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_0_2):
        for i, agent in enumerate(debate_agents_0_2):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_0_1, answer_0_1], debate_instruction_0_2, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_0_1, answer_0_1] + all_thinking_0_2[r-1] + all_answer_0_2[r-1]
                thinking, answer = await agent(input_infos, debate_instruction_0_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, constructing line arrangement, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_0_2[r].append(thinking)
            all_answer_0_2[r].append(answer)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2(
        [taskInfo, thinking_0_1, answer_0_1] + all_thinking_0_2[-1] + all_answer_0_2[-1],
        "Sub-task 2: Synthesize and finalize the line arrangement and intersection points inside the dodecagon.",
        is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 2: ", sub_tasks[-1])

    # Subtask 3: Build graph data structure of intersection points and edges (SC_CoT)
    cot_sc_instruction_0_3 = (
        "Sub-task 3: Build a graph data structure where vertices correspond to intersection points computed in Sub-task 2 and edges correspond to polygon-side or diagonal segments between these points inside the polygon. "
        "This graph represents all possible edges on which rectangle sides can lie, including partial chord segments. "
        "This step is critical to avoid the previous mistake of limiting edges to full chords between polygon vertices.")
    N_sc_0_3 = self.max_sc
    cot_agents_0_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_0_3)]
    possible_answers_0_3 = []
    possible_thinkings_0_3 = []
    subtask_desc_0_3 = {
        "subtask_id": "stage_0_subtask_3",
        "instruction": cot_sc_instruction_0_3,
        "context": ["user query", thinking_0_2.content, answer_0_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_0_3):
        thinking, answer = await cot_agents_0_3[i]([taskInfo, thinking_0_2, answer_0_2], cot_sc_instruction_0_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_3[i].id}, building intersection graph, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_3.append(answer)
        possible_thinkings_0_3.append(thinking)
    final_decision_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_3, answer_0_3 = await final_decision_agent_0_3(
        [taskInfo, thinking_0_2, answer_0_2] + possible_thinkings_0_3 + possible_answers_0_3,
        "Sub-task 3: Synthesize and finalize the intersection graph data structure.",
        is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 3: ", sub_tasks[-1])

    # Subtask 4: Analyze geometric conditions for rectangles in the graph (Debate)
    debate_instruction_0_4 = (
        "Sub-task 4: Analyze geometric conditions necessary for four edges in the graph to form a rectangle: pairs of opposite edges must be parallel and equal in length, adjacent edges must be perpendicular, and the quadrilateral formed must be convex. "
        "This subtask refines and formalizes the rectangle conditions in the context of the intersection graph, explicitly incorporating the possibility of rectangle vertices at interior intersection points rather than only polygon vertices. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.")
    debate_agents_0_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_0_4 = self.max_round
    all_thinking_0_4 = [[] for _ in range(N_max_0_4)]
    all_answer_0_4 = [[] for _ in range(N_max_0_4)]
    subtask_desc_0_4 = {
        "subtask_id": "stage_0_subtask_4",
        "instruction": debate_instruction_0_4,
        "context": ["user query", thinking_0_1.content, thinking_0_3.content, answer_0_3.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_0_4):
        for i, agent in enumerate(debate_agents_0_4):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_0_1, thinking_0_3, answer_0_3], debate_instruction_0_4, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_0_1, thinking_0_3, answer_0_3] + all_thinking_0_4[r-1] + all_answer_0_4[r-1]
                thinking, answer = await agent(input_infos, debate_instruction_0_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing rectangle conditions, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_0_4[r].append(thinking)
            all_answer_0_4[r].append(answer)
    final_decision_agent_0_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_4, answer_0_4 = await final_decision_agent_0_4(
        [taskInfo, thinking_0_1, thinking_0_3, answer_0_3] + all_thinking_0_4[-1] + all_answer_0_4[-1],
        "Sub-task 4: Synthesize and finalize the geometric rectangle conditions in the intersection graph.",
        is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_0_4.content}; answer - {answer_0_4.content}")
    subtask_desc_0_4['response'] = {"thinking": thinking_0_4, "answer": answer_0_4}
    logs.append(subtask_desc_0_4)
    print("Step 4: ", sub_tasks[-1])

    # Stage 1: Enumeration, Verification, Deduplication, and Final Counting

    # Subtask 1: Enumerate all candidate 4-cycles in the intersection graph (Reflexion)
    reflect_inst_1_1 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_1_1 = (
        "Sub-task 1: Enumerate all candidate 4-cycles (quadrilaterals) in the intersection graph constructed in stage_0.subtask_3 that could potentially form rectangles. "
        "This enumeration must consider all cycles formed by edges corresponding to polygon sides or diagonals segments, not just full chords. "
        "This addresses the previous undercounting caused by ignoring partial chord segments and interior intersection points. "
        + reflect_inst_1_1)
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1_1 = self.max_round
    cot_inputs_1_1 = [taskInfo, thinking_0_3, answer_0_3]
    subtask_desc_1_1 = {
        "subtask_id": "stage_1_subtask_1",
        "instruction": cot_reflect_instruction_1_1,
        "context": ["user query", thinking_0_3.content, answer_0_3.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1(cot_inputs_1_1, cot_reflect_instruction_1_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_1.id}, enumerating candidate quadrilaterals, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    for i in range(N_max_1_1):
        feedback, correct = await critic_agent_1_1([taskInfo, thinking_1_1, answer_1_1],
                                                 "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'",
                                                 i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_1.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_1_1.extend([thinking_1_1, answer_1_1, feedback])
        thinking_1_1, answer_1_1 = await cot_agent_1_1(cot_inputs_1_1, cot_reflect_instruction_1_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_1.id}, refining enumeration, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 5: ", sub_tasks[-1])

    # Subtask 2: Verify rectangle properties for each candidate quadrilateral (SC_CoT)
    cot_sc_instruction_1_2 = (
        "Sub-task 2: For each candidate quadrilateral from Sub-task 1, rigorously verify rectangle properties: check that opposite sides are parallel and equal in length, adjacent sides are perpendicular (using vector dot products), vertices are distinct, and the quadrilateral is convex. "
        "This verification must be explicit and numerical/algebraic to avoid previous errors relying on heuristic or incomplete checks.")
    N_sc_1_2 = self.max_sc
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_2)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1_subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_1_1.content, answer_1_1.content, thinking_0_4.content, answer_0_4.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_2):
        thinking, answer = await cot_agents_1_2[i]([taskInfo, thinking_1_1, answer_1_1, thinking_0_4, answer_0_4], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, verifying rectangle properties, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_1_2.append(answer)
        possible_thinkings_1_2.append(thinking)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2(
        [taskInfo, thinking_1_1, answer_1_1, thinking_0_4, answer_0_4] + possible_thinkings_1_2 + possible_answers_1_2,
        "Sub-task 2: Synthesize and finalize verification of rectangle properties.",
        is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 6: ", sub_tasks[-1])

    # Subtask 3: Identify and eliminate duplicate rectangles due to symmetries (Debate)
    debate_instruction_1_3 = (
        "Sub-task 3: Implement a robust method to identify and eliminate duplicate rectangles arising from the polygon's rotational and reflection symmetries. "
        "Maintain a data structure to track unique rectangles up to these symmetries, ensuring no overcounting. "
        "This step addresses the previous failure to handle symmetry rigorously, which led to incorrect final counts. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.")
    debate_agents_1_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1_3 = self.max_round
    all_thinking_1_3 = [[] for _ in range(N_max_1_3)]
    all_answer_1_3 = [[] for _ in range(N_max_1_3)]
    subtask_desc_1_3 = {
        "subtask_id": "stage_1_subtask_3",
        "instruction": debate_instruction_1_3,
        "context": ["user query", thinking_1_2.content, answer_1_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_3):
        for i, agent in enumerate(debate_agents_1_3):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_1_2, answer_1_2], debate_instruction_1_3, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_1_2, answer_1_2] + all_thinking_1_3[r-1] + all_answer_1_3[r-1]
                thinking, answer = await agent(input_infos, debate_instruction_1_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, eliminating duplicates, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_1_3[r].append(thinking)
            all_answer_1_3[r].append(answer)
    final_decision_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await final_decision_agent_1_3(
        [taskInfo, thinking_1_2, answer_1_2] + all_thinking_1_3[-1] + all_answer_1_3[-1],
        "Sub-task 3: Synthesize and finalize unique rectangles after symmetry elimination.",
        is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    subtask_desc_1_3['response'] = {"thinking": thinking_1_3, "answer": answer_1_3}
    logs.append(subtask_desc_1_3)
    print("Step 7: ", sub_tasks[-1])

    # Subtask 4: Aggregate verified unique rectangles and derive final count (Debate)
    debate_instruction_1_4 = (
        "Sub-task 4: Aggregate the verified unique rectangles and derive a formal count. Where possible, develop a combinatorial or geometric proof to confirm the correctness of the enumeration and counting. "
        "Summarize the final number of rectangles formed inside the dodecagon with sides on its sides or diagonals, explicitly including those formed by partial chord segments and interior intersection points. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.")
    debate_agents_1_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1_4 = self.max_round
    all_thinking_1_4 = [[] for _ in range(N_max_1_4)]
    all_answer_1_4 = [[] for _ in range(N_max_1_4)]
    subtask_desc_1_4 = {
        "subtask_id": "stage_1_subtask_4",
        "instruction": debate_instruction_1_4,
        "context": ["user query", thinking_1_3.content, answer_1_3.content, thinking_0_1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_4):
        for i, agent in enumerate(debate_agents_1_4):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_1_3, answer_1_3, thinking_0_1], debate_instruction_1_4, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_1_3, answer_1_3, thinking_0_1] + all_thinking_1_4[r-1] + all_answer_1_4[r-1]
                thinking, answer = await agent(input_infos, debate_instruction_1_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, aggregating final count, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_1_4[r].append(thinking)
            all_answer_1_4[r].append(answer)
    final_decision_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_4, answer_1_4 = await final_decision_agent_1_4(
        [taskInfo, thinking_1_3, answer_1_3, thinking_0_1] + all_thinking_1_4[-1] + all_answer_1_4[-1],
        "Sub-task 4: Synthesize and finalize the count of rectangles inside the dodecagon.",
        is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_1_4.content}; answer - {answer_1_4.content}")
    subtask_desc_1_4['response'] = {"thinking": thinking_1_4, "answer": answer_1_4}
    logs.append(subtask_desc_1_4)
    print("Step 8: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_1_4, answer_1_4, sub_tasks, agents)
    return final_answer, logs
