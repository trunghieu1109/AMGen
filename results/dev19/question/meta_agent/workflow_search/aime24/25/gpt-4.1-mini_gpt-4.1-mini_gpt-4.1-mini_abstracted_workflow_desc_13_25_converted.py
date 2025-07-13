async def forward_25(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_0_1 = "Sub-task 1: Formally define vector representations for the sides of the hexagon ABCDEF, incorporating the conditions that the hexagon is equilateral and that opposite sides are parallel. Explicitly express the directions and magnitudes of sides AB, BC, CD, DE, EF, and FA as vectors, introducing variables for unknown angles or directions as needed. Avoid assuming specific numeric values at this stage."
    N_sc = self.max_sc
    cot_agents_0_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_0_1 = []
    possible_thinkings_0_1 = []
    subtask_desc_0_1 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": cot_sc_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking, answer = await cot_agents_0_1[i]([taskInfo], cot_sc_instruction_0_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_1[i].id}, defining vector representations, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_1.append(answer)
        possible_thinkings_0_1.append(thinking)
    final_decision_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_1, answer_0_1 = await final_decision_agent_0_1([taskInfo] + possible_thinkings_0_1, "Sub-task 1: Synthesize vector definitions for hexagon sides." , is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_0.subtask_1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step stage_0.subtask_1: ", sub_tasks[-1])

    cot_sc_instruction_0_2 = "Sub-task 2: Derive the geometric relationships and constraints imposed by the hexagon's convexity, equilateral property, and parallelism of opposite sides on the vectors defined in subtask_1. This includes vector sum conditions and angle restrictions to ensure convexity. Avoid attempting to solve these constraints numerically yet."
    cot_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "stage_0.subtask_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking, answer = await cot_agents_0_2[i]([taskInfo, thinking_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, deriving geometric constraints, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_2.append(answer)
        possible_thinkings_0_2.append(thinking)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2([taskInfo] + possible_thinkings_0_2, "Sub-task 2: Synthesize geometric constraints for hexagon vectors.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_0.subtask_2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step stage_0.subtask_2: ", sub_tasks[-1])

    cot_sc_instruction_0_3 = "Sub-task 3: Formulate the construction of the triangle formed by the extensions of sides AB, CD, and EF. Define the intersection points of these extended lines and express the triangle's side lengths in terms of the hexagon's side vectors and their directions. Clarify assumptions about the order and orientation of these extensions to avoid ambiguity."
    cot_agents_0_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_0_3 = []
    possible_thinkings_0_3 = []
    subtask_desc_0_3 = {
        "subtask_id": "stage_0.subtask_3",
        "instruction": cot_sc_instruction_0_3,
        "context": ["user query", thinking_0_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking, answer = await cot_agents_0_3[i]([taskInfo, thinking_0_1], cot_sc_instruction_0_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_3[i].id}, formulating triangle construction, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_3.append(answer)
        possible_thinkings_0_3.append(thinking)
    final_decision_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_3, answer_0_3 = await final_decision_agent_0_3([taskInfo] + possible_thinkings_0_3, "Sub-task 3: Synthesize triangle construction and side length expressions.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_0.subtask_3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step stage_0.subtask_3: ", sub_tasks[-1])

    reflect_inst_1_1 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_1_1 = "Sub-task 1: Combine the vector representations and geometric constraints from stage_0 to derive explicit formulas relating the side length of the hexagon to the side lengths of the triangle formed by the extensions of AB, CD, and EF. This involves expressing the triangle's side lengths (200, 240, 300) as functions of the hexagon's side length and direction angles." + reflect_inst_1_1
    cot_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1_1 = self.max_round
    cot_inputs_1_1 = [taskInfo, thinking_0_2, thinking_0_3]
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_reflect_instruction_1_1,
        "context": ["user query", thinking_0_2.content, thinking_0_3.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_1, answer_1_1 = await cot_agent_1_1(cot_inputs_1_1, cot_reflect_instruction_1_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_1.id}, deriving formulas, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    for i in range(N_max_1_1):
        feedback, correct = await critic_agent_1_1([taskInfo, thinking_1_1], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_1.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_1_1.extend([thinking_1_1, feedback])
        thinking_1_1, answer_1_1 = await cot_agent_1_1(cot_inputs_1_1, cot_reflect_instruction_1_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_1.id}, refining formulas, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task stage_1.subtask_1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step stage_1.subtask_1: ", sub_tasks[-1])

    reflect_inst_1_2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_1_2 = "Sub-task 2: Transform and simplify the derived formulas to isolate the hexagon's side length as a variable, preparing the system of equations for solving. Identify any additional unknown parameters (e.g., angles) that must be determined or constrained to solve for the side length." + reflect_inst_1_2
    cot_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1_2 = self.max_round
    cot_inputs_1_2 = [taskInfo, thinking_1_1]
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_reflect_instruction_1_2,
        "context": ["user query", thinking_1_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking_1_2, answer_1_2 = await cot_agent_1_2(cot_inputs_1_2, cot_reflect_instruction_1_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1_2.id}, simplifying formulas, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    for i in range(N_max_1_2):
        feedback, correct = await critic_agent_1_2([taskInfo, thinking_1_2], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1_2.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_1_2.extend([thinking_1_2, feedback])
        thinking_1_2, answer_1_2 = await cot_agent_1_2(cot_inputs_1_2, cot_reflect_instruction_1_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1_2.id}, refining simplification, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Sub-task stage_1.subtask_2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step stage_1.subtask_2: ", sub_tasks[-1])

    cot_sc_instruction_2_1 = "Sub-task 1: Analyze the system of equations obtained in stage_1 to infer the values of unknown parameters, including the hexagon's side length. Use algebraic or geometric methods to solve the system under the given constraints, ensuring solutions are consistent with convexity and parallelism conditions."
    cot_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    subtask_desc_2_1 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_sc_instruction_2_1,
        "context": ["user query", thinking_1_2.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking, answer = await cot_agents_2_1[i]([taskInfo, thinking_1_2], cot_sc_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_1[i].id}, solving system, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_2_1.append(answer)
        possible_thinkings_2_1.append(thinking)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_1, answer_2_1 = await final_decision_agent_2_1([taskInfo] + possible_thinkings_2_1, "Sub-task 1: Synthesize solution for unknown parameters.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_2.subtask_1 output: thinking - {thinking_2_1.content}; answer - {answer_2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking_2_1, "answer": answer_2_1}
    logs.append(subtask_desc_2_1)
    print("Step stage_2.subtask_1: ", sub_tasks[-1])

    cot_sc_instruction_2_2 = "Sub-task 2: Compute the numerical value of the hexagon's side length from the solved parameters, ensuring the result matches the given triangle side lengths and the hexagon's properties."
    cot_agents_2_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_2_2 = []
    possible_thinkings_2_2 = []
    subtask_desc_2_2 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_sc_instruction_2_2,
        "context": ["user query", thinking_2_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking, answer = await cot_agents_2_2[i]([taskInfo, thinking_2_1], cot_sc_instruction_2_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2_2[i].id}, computing numerical side length, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_2_2.append(answer)
        possible_thinkings_2_2.append(thinking)
    final_decision_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_2, answer_2_2 = await final_decision_agent_2_2([taskInfo] + possible_thinkings_2_2, "Sub-task 2: Synthesize numerical hexagon side length.", is_sub_task=True)
    sub_tasks.append(f"Sub-task stage_2.subtask_2 output: thinking - {thinking_2_2.content}; answer - {answer_2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking_2_2, "answer": answer_2_2}
    logs.append(subtask_desc_2_2)
    print("Step stage_2.subtask_2: ", sub_tasks[-1])

    debate_instr_3_1 = "Sub-task 1: Verify that the computed hexagon side length and associated parameters satisfy all problem constraints, including convexity, equilateral property, and parallelism of opposite sides. Confirm that the triangle formed by the extensions has side lengths exactly 200, 240, and 300 as given. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_3_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_3_1 = self.max_round
    all_thinking_3_1 = [[] for _ in range(N_max_3_1)]
    all_answer_3_1 = [[] for _ in range(N_max_3_1)]
    subtask_desc_3_1 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instr_3_1,
        "context": ["user query", thinking_2_2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3_1):
        for i, agent in enumerate(debate_agents_3_1):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_2_2], debate_instr_3_1, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_2_2] + all_thinking_3_1[r-1]
                thinking, answer = await agent(input_infos, debate_instr_3_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying solution, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_3_1[r].append(thinking)
            all_answer_3_1[r].append(answer)
    final_decision_agent_3_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_decision_instruction_3_1 = "Sub-task 1: Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking_3_1, answer_3_1 = await final_decision_agent_3_1([taskInfo] + all_thinking_3_1[-1], final_decision_instruction_3_1, is_sub_task=True)
    agents.append(f"Final Decision agent, verifying solution, thinking: {thinking_3_1.content}; answer: {answer_3_1.content}")
    sub_tasks.append(f"Sub-task stage_3.subtask_1 output: thinking - {thinking_3_1.content}; answer - {answer_3_1.content}")
    subtask_desc_3_1['response'] = {"thinking": thinking_3_1, "answer": answer_3_1}
    logs.append(subtask_desc_3_1)
    print("Step stage_3.subtask_1: ", sub_tasks[-1])

    debate_instr_3_2 = "Sub-task 2: Confirm the uniqueness or identify possible alternative solutions for the hexagon's side length under the given constraints, discussing the geometric implications and excluding invalid configurations. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_3_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_3_2 = self.max_round
    all_thinking_3_2 = [[] for _ in range(N_max_3_2)]
    all_answer_3_2 = [[] for _ in range(N_max_3_2)]
    subtask_desc_3_2 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": debate_instr_3_2,
        "context": ["user query", thinking_3_1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3_2):
        for i, agent in enumerate(debate_agents_3_2):
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking_3_1], debate_instr_3_2, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking_3_1] + all_thinking_3_2[r-1]
                thinking, answer = await agent(input_infos, debate_instr_3_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, confirming uniqueness, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking_3_2[r].append(thinking)
            all_answer_3_2[r].append(answer)
    final_decision_agent_3_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_decision_instruction_3_2 = "Sub-task 2: Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking_3_2, answer_3_2 = await final_decision_agent_3_2([taskInfo] + all_thinking_3_2[-1], final_decision_instruction_3_2, is_sub_task=True)
    agents.append(f"Final Decision agent, confirming uniqueness, thinking: {thinking_3_2.content}; answer: {answer_3_2.content}")
    sub_tasks.append(f"Sub-task stage_3.subtask_2 output: thinking - {thinking_3_2.content}; answer - {answer_3_2.content}")
    subtask_desc_3_2['response'] = {"thinking": thinking_3_2, "answer": answer_3_2}
    logs.append(subtask_desc_3_2)
    print("Step stage_3.subtask_2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_2, answer_3_2, sub_tasks, agents)
    return final_answer, logs
