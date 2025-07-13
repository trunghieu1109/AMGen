async def forward_21(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Domain and foundational properties

    # Subtask 1: Identify and clearly state the domain of the problem by defining the set of all chords
    cot_sc_instruction_0_1 = (
        "Sub-task 1: Identify and clearly state the domain of the problem by defining the set of all chords "
        "(including sides and diagonals) of a fixed regular dodecagon. Describe their geometric properties, "
        "how they relate to the polygon's vertices, and the significance of chord step sizes modulo 12. "
        "Avoid attempting to solve any perpendicularity or rectangle conditions at this stage.")
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
        agents.append(f"CoT-SC agent {cot_agents_0_1[i].id}, defining chords domain, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_1.append(answer)
        possible_thinkings_0_1.append(thinking)
    final_decision_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_1, answer_0_1 = await final_decision_agent_0_1([taskInfo] + possible_thinkings_0_1, "Sub-task 1: Synthesize and choose the most consistent answer for chord domain definition.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_0.subtask_1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step stage_0.subtask_1: ", sub_tasks[-1])

    # Subtask 2: Enumerate and classify all chords into sides and diagonals
    cot_sc_instruction_0_2 = (
        "Sub-task 2: Enumerate and classify all chords of the dodecagon into sides and diagonals, "
        "specifying their counts and relevant properties such as length and orientation. Emphasize the combinatorial structure of chords and their indexing by vertex step sizes. "
        "Do not yet consider perpendicularity or rectangle formation.")
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
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, enumerating chords, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_2.append(answer)
        possible_thinkings_0_2.append(thinking)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo, thinking_0_1] + possible_thinkings_0_2, "Sub-task 2: Synthesize and choose the most consistent answer for chord enumeration.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_0.subtask_2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step stage_0.subtask_2: ", sub_tasks[-1])

    # Subtask 3: Define precisely what it means for rectangle sides to lie on sides or diagonals
    debate_instruction_0_3 = (
        "Sub-task 3: Define precisely what it means for a rectangle side to lie on a side or diagonal of the dodecagon. "
        "Clarify assumptions that rectangles must have vertices at polygon vertices, that rectangle sides coincide exactly with polygon chords, and that rectangles are inscribed inside the polygon. Avoid ambiguity about vertex placement or side definitions. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.")
    debate_agents_0_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_0_3 = self.max_round
    all_thinking_0_3 = [[] for _ in range(N_max_0_3)]
    all_answer_0_3 = [[] for _ in range(N_max_0_3)]
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": debate_instruction_0_3,
        "context": ["user query", thinking_0_1.content, thinking_0_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_0_3):
        for i, agent in enumerate(debate_agents_0_3):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_0_1, thinking_0_2], debate_instruction_0_3, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_0_1, thinking_0_2] + all_thinking_0_3[r-1]
                thinking, answer = await agent(input_infos, debate_instruction_0_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, defining rectangle side conditions, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_0_3[r].append(thinking)
            all_answer_0_3[r].append(answer)
    final_decision_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_3, answer_0_3 = await final_decision_agent_0_3([taskInfo, thinking_0_1, thinking_0_2] + all_thinking_0_3[-1], "Sub-task 3: Synthesize and finalize rectangle side definition.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_0.subtask_3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step stage_0.subtask_3: ", sub_tasks[-1])

    # Subtask 4: Derive rigorously the correct geometric condition for perpendicularity
    debate_instruction_0_4 = (
        "Sub-task 4: Derive rigorously from first principles the correct geometric condition for two chords of the dodecagon to be perpendicular. "
        "Prove that two chords with step sizes k and m satisfy perpendicularity if and only if k + m ≡ 6 (mod 12). Use vector or complex number representations of vertices and chord orientations. "
        "Do not proceed to enumeration until this proof is accepted and verified. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.")
    debate_agents_0_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_0_4 = self.max_round
    all_thinking_0_4 = [[] for _ in range(N_max_0_4)]
    all_answer_0_4 = [[] for _ in range(N_max_0_4)]
    subtask_desc_0_4 = {
        "subtask_id": "stage_0.subtask_4",
        "instruction": debate_instruction_0_4,
        "context": ["user query", thinking_0_1.content, thinking_0_2.content, thinking_0_3.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_0_4):
        for i, agent in enumerate(debate_agents_0_4):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_0_1, thinking_0_2, thinking_0_3], debate_instruction_0_4, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_0_1, thinking_0_2, thinking_0_3] + all_thinking_0_4[r-1]
                thinking, answer = await agent(input_infos, debate_instruction_0_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, proving perpendicularity condition, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_0_4[r].append(thinking)
            all_answer_0_4[r].append(answer)
    final_decision_agent_0_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_4, answer_0_4 = await final_decision_agent_0_4([taskInfo, thinking_0_1, thinking_0_2, thinking_0_3] + all_thinking_0_4[-1], "Sub-task 4: Synthesize and finalize perpendicularity proof.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_0.subtask_4 output: thinking - {thinking_0_4.content}; answer - {answer_0_4.content}")
    subtask_desc_0_4['response'] = {"thinking": thinking_0_4, "answer": answer_0_4}
    logs.append(subtask_desc_0_4)
    print("Step stage_0.subtask_4: ", sub_tasks[-1])

    # Stage 1: Characterize rectangles using the verified perpendicularity condition

    # Subtask 1: Characterize sets of four vertices forming rectangles
    debate_instruction_1_1 = (
        "Sub-task 1: Using the verified perpendicularity condition (k + m = 6 mod 12), characterize the sets of four vertices (V_a, V_{a+k}, V_{a+k+6}, V_{a+6}) that form rectangles inside the dodecagon. "
        "Derive the necessary and sufficient conditions for these quadruples to form rectangles with sides on polygon chords. Avoid using any incorrect perpendicularity assumptions. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.")
    debate_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_1_1 = self.max_round
    all_thinking_1_1 = [[] for _ in range(N_max_1_1)]
    all_answer_1_1 = [[] for _ in range(N_max_1_1)]
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": debate_instruction_1_1,
        "context": ["user query", thinking_0_4.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_1):
        for i, agent in enumerate(debate_agents_1_1):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_0_4], debate_instruction_1_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_0_4] + all_thinking_1_1[r-1]
                thinking, answer = await agent(input_infos, debate_instruction_1_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, characterizing rectangles, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_1_1[r].append(thinking)
            all_answer_1_1[r].append(answer)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1([taskInfo, thinking_0_4] + all_thinking_1_1[-1], "Sub-task 1: Synthesize and finalize rectangle characterization.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_1.subtask_1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1 = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append({"subtask_id": "stage_1.subtask_1", "response": subtask_desc_1_1})
    print("Step stage_1.subtask_1: ", sub_tasks[-1])

    # Subtask 2: Validate rectangles have vertices exactly at polygon vertices and sides coincide with chords
    cot_sc_instruction_1_2 = (
        "Sub-task 2: Validate that all rectangles formed under the derived conditions have vertices exactly at polygon vertices and that their sides correspond exactly to polygon sides or diagonals. "
        "Rule out rectangles with vertices inside edges or on non-vertex points. Confirm that the rectangle sides coincide with polygon chords as defined.")
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(self.max_sc)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking_0_4.content, thinking_1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking, answer = await cot_agents_1_2[i]([taskInfo, thinking_0_4, thinking_1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, validating rectangle vertices and sides, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_1_2.append(answer)
        possible_thinkings_1_2.append(thinking)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo, thinking_0_4, thinking_1_1] + possible_thinkings_1_2, "Sub-task 2: Synthesize and finalize validation of rectangle vertices and sides.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_1.subtask_2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step stage_1.subtask_2: ", sub_tasks[-1])

    # Stage 2: Enumeration and symmetry classification

    # Subtask 1: Enumerate all possible quadruples forming rectangles
    debate_instruction_2_1 = (
        "Sub-task 1: Enumerate all possible quadruples of vertices forming rectangles inside the dodecagon under the corrected perpendicularity condition and vertex constraints. "
        "Use the polygon's rotational symmetry to generate candidates systematically. Include small-scale validation checks on polygons with fewer sides (e.g., n=4 or n=8) to confirm correctness of enumeration before full enumeration for n=12. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.")
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_2_1 = self.max_round
    all_thinking_2_1 = [[] for _ in range(N_max_2_1)]
    all_answer_2_1 = [[] for _ in range(N_max_2_1)]
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": debate_instruction_2_1,
        "context": ["user query", thinking_1_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_1):
        for i, agent in enumerate(debate_agents_2_1):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_1_2], debate_instruction_2_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_1_2] + all_thinking_2_1[r-1]
                thinking, answer = await agent(input_infos, debate_instruction_2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, enumerating rectangles, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_2_1[r].append(thinking)
            all_answer_2_1[r].append(answer)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo, thinking_1_2] + all_thinking_2_1[-1], "Sub-task 1: Synthesize and finalize enumeration.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_2.subtask_1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1 = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append({"subtask_id": "stage_2.subtask_1", "response": subtask_desc_2_1})
    print("Step stage_2.subtask_1: ", sub_tasks[-1])

    # Subtask 2: Analyze enumerated rectangles to identify equivalence classes under D_12
    cot_sc_instruction_2_2 = (
        "Sub-task 2: Analyze the enumerated rectangles to identify equivalence classes under the dihedral symmetry group D_12 (rotations and reflections). "
        "Classify rectangles into distinct symmetry classes to avoid overcounting. Provide explicit reasoning and counts of these classes.")
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
        agents.append(f"CoT-SC agent {cot_agents_2_2[i].id}, analyzing symmetry classes, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_2_2.append(answer)
        possible_thinkings_2_2.append(thinking)
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_2, answer_2_2 = await final_decision_agent_2_2([taskInfo, thinking_2_1] + possible_thinkings_2_2, "Sub-task 2: Synthesize and finalize symmetry classification.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_2.subtask_2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step stage_2.subtask_2: ", sub_tasks[-1])

    # Subtask 3: Summarize symmetry classification results
    reflect_instruction_2_3 = (
        "Sub-task 3: Summarize the symmetry classification results, explicitly listing the number of distinct rectangle classes and their representatives. "
        "Provide a clear bridge between enumeration and final counting, ensuring all symmetry considerations are transparent and well-documented. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better.")
    cot_agent_2_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2_3 = self.max_round
    cot_inputs_2_3 = [taskInfo, thinking_2_1, thinking_2_2]
    subtask_desc_2_3 = {
        "subtask_id": "stage_2.subtask_3",
        "instruction": reflect_instruction_2_3,
        "context": ["user query", thinking_2_1.content, thinking_2_2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_2_3, answer_2_3 = await cot_agent_2_3(cot_inputs_2_3, reflect_instruction_2_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_3.id}, summarizing symmetry classes, thinking: {thinking_2_3.content}; answer: {answer_2_3.content}")
    for i in range(N_max_2_3):
        feedback, correct = await critic_agent_2_3([taskInfo, thinking_2_3], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2_3.extend([thinking_2_3, feedback])
        thinking_2_3, answer_2_3 = await cot_agent_2_3(cot_inputs_2_3, reflect_instruction_2_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_3.id}, refining symmetry summary, thinking: {thinking_2_3.content}; answer: {answer_2_3.content}")
    sub_tasks.append(f"Sub-task stage_2.subtask_3 output: thinking - {thinking_2_3.content}; answer - {answer_2_3.content}")
    subtask_desc_2_3['response'] = {"thinking": thinking_2_3, "answer": answer_2_3}
    logs.append(subtask_desc_2_3)
    print("Step stage_2.subtask_3: ", sub_tasks[-1])

    # Stage 3: Final counting and summation

    # Subtask 1: Decompose total count based on orientation/type using symmetry classes
    cot_instruction_3_1 = (
        "Sub-task 1: Decompose the total count of rectangles into components based on their orientation or type (e.g., axis-aligned vs. tilted rectangles) using the symmetry classes identified. "
        "Simplify counting expressions to minimal form, ensuring no double counting occurs.")
    cot_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": cot_instruction_3_1,
        "context": ["user query", thinking_2_3.content],
        "agent_collaboration": "CoT"
    }
    thinking_3_1, answer_3_1 = await cot_agent_3_1([taskInfo, thinking_2_3], cot_instruction_3_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3_1.id}, decomposing rectangle counts, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    sub_tasks.append(f"Sub-task stage_3.subtask_1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step stage_3.subtask_1: ", sub_tasks[-1])

    # Subtask 2: Sum all components to obtain final total number of distinct rectangles
    reflect_instruction_3_2 = (
        "Sub-task 2: Sum all components from the decomposition to obtain the final total number of distinct rectangles formed inside the regular dodecagon with sides on its sides or diagonals. "
        "Explicitly incorporate the symmetry reduction results to avoid overcounting. Present the final answer with clear justification referencing all prior results. "
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better.")
    cot_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3_2 = self.max_round
    cot_inputs_3_2 = [taskInfo, thinking_2_3, thinking_3_1]
    subtask_desc_3_2 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": reflect_instruction_3_2,
        "context": ["user query", thinking_2_3.content, thinking_3_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_3_2, answer_3_2 = await cot_agent_3_2(cot_inputs_3_2, reflect_instruction_3_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3_2.id}, summing final counts, thinking: {thinking_3_2.content}; answer: {answer_3_2.content}")
    for i in range(N_max_3_2):
        feedback, correct = await critic_agent_3_2([taskInfo, thinking_3_2], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3_2.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3_2.extend([thinking_3_2, feedback])
        thinking_3_2, answer_3_2 = await cot_agent_3_2(cot_inputs_3_2, reflect_instruction_3_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3_2.id}, refining final count, thinking: {thinking_3_2.content}; answer: {answer_3_2.content}")
    sub_tasks.append(f"Sub-task stage_3.subtask_2 output: thinking - {thinking_3_2.content}; answer - {answer_3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking_3_2, "answer": answer_3_2}
    logs.append(subtask_desc_3_2)
    print("Step stage_3.subtask_2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_2, answer_3_2, sub_tasks, agents)
    return final_answer, logs
