async def forward_16(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Coordinate Setup and Geometric Constraints
    # Subtask 1: Represent O, I, A and formulate perpendicularity IA ⟂ OI using CoT
    cot_instruction_1 = (
        "Sub-task 1: Formally represent the given elements and constraints using coordinate/vector geometry. "
        "Place circumcenter O at origin. Express incenter I with unknown coordinates constrained by Euler's formula OI^2 = R(R - 2r). "
        "Represent vertex A on circumcircle |OA|=13 with unknown coordinates. Formulate IA ⟂ OI as dot product zero. "
        "Derive algebraic expressions for these constraints without assuming special triangle types."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, representing coordinates and perpendicularity, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)

    # Subtask 2: Formulate and enforce inradius condition (distance from I to BC = 6) using CoT
    cot_instruction_2 = (
        "Sub-task 2: Express the inradius condition explicitly: the distance from incenter I to side BC equals 6. "
        "Represent side BC in terms of unknown parameters or vectors consistent with previous subtask outputs. "
        "Derive an algebraic equation enforcing this distance condition, providing a second independent constraint. "
        "Use outputs from Sub-task 1 and introduce variables for B and C as needed, avoiding assumptions about triangle shape."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, formulating inradius condition, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)

    # Stage 2: Geometric Verification and Side Length Expressions
    # Subtask 1: Verify geometric interpretations of vectors and angles using Reflexion
    reflect_instruction_1 = (
        "Sub-task 1: Analyze and verify whether the angle between vectors OA and OI corresponds to the half-angle at vertex A or any meaningful triangle angle. "
        "Avoid assuming AB=AC or isosceles properties without proof. Use geometric reasoning or algebraic checks to validate angle correspondences. "
        "If invalid, revise approach accordingly to prevent propagation of incorrect assumptions."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": reflect_instruction_1,
        "context": ["user query", thinking1.content, answer1.content, thinking2.content, answer2.content],
        "agent_collaboration": "Reflexion"
    }
    cot_inputs_3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, reflect_instruction_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, verifying geometric angle interpretations, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent_1([taskInfo, thinking3, answer3],
                                               "Please review and provide limitations of the solution. If correct, output exactly 'True' in 'correct'",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, reflect_instruction_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining verification, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)

    # Subtask 2: Derive expressions for sides AB and AC using SC_CoT
    cot_sc_instruction_1 = (
        "Sub-task 2: Using verified geometric relations and constraints, derive expressions for sides AB and AC. "
        "Apply Law of Cosines or classical triangle relations involving sides and angles at A. "
        "Express AB and AC in terms of known parameters (R=13, r=6) and variables from previous subtasks. "
        "Incorporate perpendicularity and inradius conditions explicitly. Avoid new assumptions; rely on verified relations. "
        "Prepare formulas for computing product AB·AC."
    )
    N_sc = self.max_sc
    cot_agents_sc = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    subtask_desc4 = {
        "subtask_id": "stage_2.subtask_2",
        "instruction": cot_sc_instruction_1,
        "context": ["user query", thinking1.content, answer1.content, thinking2.content, answer2.content, thinking3.content, answer3.content],
        "agent_collaboration": "SC_CoT"
    }
    possible_answers = []
    possible_thinkings = []
    for i in range(N_sc):
        thinking_sc, answer_sc = await cot_agents_sc[i]([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc[i].id}, deriving sides AB and AC, thinking: {thinking_sc.content}; answer: {answer_sc.content}")
        possible_answers.append(answer_sc)
        possible_thinkings.append(thinking_sc)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_2([taskInfo] + possible_answers + possible_thinkings,
                                                    "Sub-task 2: Synthesize and choose the most consistent and correct expressions for sides AB and AC.",
                                                    is_sub_task=True)
    agents.append(f"Final Decision agent, synthesizing sides AB and AC, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)

    # Stage 3: Numeric Solution and Verification
    # Subtask 1: Solve system and compute AB·AC using Debate
    debate_instr_3 = (
        "Sub-task 1: Solve the system of equations from perpendicularity and inradius conditions numerically. "
        "Calculate numeric values of sides AB and AC and compute product AB·AC explicitly. "
        "Ensure all constraints are satisfied. Avoid rounding errors or premature approximations. "
        "Given solutions from other agents, consider their opinions as advice and provide updated answers."
    )
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking3 = [[] for _ in range(N_max_3)]
    all_answer3 = [[] for _ in range(N_max_3)]
    subtask_desc5 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instr_3,
        "context": ["user query", thinking4.content, answer4.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking_d, answer_d = await agent([taskInfo, thinking4, answer4], debate_instr_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking4, answer4] + all_thinking3[r-1] + all_answer3[r-1]
                thinking_d, answer_d = await agent(input_infos_3, debate_instr_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, solving system and computing AB·AC, thinking: {thinking_d.content}; answer: {answer_d.content}")
            all_thinking3[r].append(thinking_d)
            all_answer3[r].append(answer_d)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_3([taskInfo] + all_thinking3[-1] + all_answer3[-1],
                                                    "Sub-task 1: Provide final numeric solution for AB·AC.",
                                                    is_sub_task=True)
    agents.append(f"Final Decision agent, final numeric solution, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)

    # Subtask 2: Verify computed solution using Reflexion
    reflect_instruction_2 = (
        "Sub-task 2: Verify that the computed solution satisfies all geometric conditions numerically: "
        "IA ⟂ OI, distance from I to BC equals 6, |OA|=13, and triangle validity. "
        "Identify and report any inconsistencies for correction."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "stage_3.subtask_2",
        "instruction": reflect_instruction_2,
        "context": ["user query", thinking5.content, answer5.content],
        "agent_collaboration": "Reflexion"
    }
    cot_inputs_4 = [taskInfo, thinking5, answer5]
    thinking6, answer6 = await cot_agent_4(cot_inputs_4, reflect_instruction_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, verifying numeric solution, thinking: {thinking6.content}; answer: {answer6.content}")
    for i in range(self.max_round):
        feedback2, correct2 = await critic_agent_2([taskInfo, thinking6, answer6],
                                                 "Please review and provide limitations of the numeric solution. If correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2.id}, feedback: {feedback2.content}; correct: {correct2.content}")
        if correct2.content == "True":
            break
        cot_inputs_4.extend([thinking6, answer6, feedback2])
        thinking6, answer6 = await cot_agent_4(cot_inputs_4, reflect_instruction_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining verification, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)

    # Stage 4: Summary and Final Answer
    # Subtask 1: Summarize solution and present final answer using SC_CoT
    cot_sc_instruction_2 = (
        "Sub-task 1: Summarize the entire solution process, constraints enforced, and verification performed. "
        "Present the numeric value of AB·AC as the final answer with justification based on verification. "
        "Highlight that all geometric conditions have been rigorously enforced and checked. "
        "Avoid new computations or assumptions."
    )
    cot_agents_sc_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc7 = {
        "subtask_id": "stage_4.subtask_1",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking6.content, answer6.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking_sc2, answer_sc2 = await cot_agents_sc_2[i]([taskInfo, thinking6, answer6], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_sc_2[i].id}, summarizing solution, thinking: {thinking_sc2.content}; answer: {answer_sc2.content}")
        possible_answers_2.append(answer_sc2)
        possible_thinkings_2.append(thinking_sc2)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_4([taskInfo] + possible_answers_2 + possible_thinkings_2,
                                                    "Sub-task 1: Provide final summarized answer for AB·AC.",
                                                    is_sub_task=True)
    agents.append(f"Final Decision agent, final summary and answer, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)

    print("Step 1: ", sub_tasks[0])
    print("Step 2: ", sub_tasks[1])
    print("Step 3: ", sub_tasks[2])
    print("Step 4: ", sub_tasks[3])
    print("Step 5: ", sub_tasks[4])
    print("Step 6: ", sub_tasks[5])
    print("Step 7: ", sub_tasks[6])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer, logs
