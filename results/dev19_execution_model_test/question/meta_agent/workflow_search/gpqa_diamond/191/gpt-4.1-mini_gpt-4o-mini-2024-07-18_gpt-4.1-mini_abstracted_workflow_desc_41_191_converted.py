async def forward_191(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_instr_stage1 = "Sub-task 1: Extract and clearly define all given physical parameters, geometric configurations, and vector relationships from the problem statement, including the conductor radius R, cavity radius r and displacement s, charge +q location inside the cavity, observation point P coordinates, distances L and l, and the angle theta between vectors. Explicitly identify the need to consider the conductor's induced charges and the off-center cavity displacement to avoid premature assumptions about field symmetry. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_stage1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_stage1 = self.max_round
    all_thinking_stage1 = [[] for _ in range(N_max_stage1)]
    all_answer_stage1 = [[] for _ in range(N_max_stage1)]
    subtask_desc1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": debate_instr_stage1,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_stage1):
        for i, agent in enumerate(debate_agents_stage1):
            if r == 0:
                thinking1, answer1 = await agent([taskInfo], debate_instr_stage1, r, is_sub_task=True)
            else:
                input_infos_1 = [taskInfo] + all_thinking_stage1[r-1] + all_answer_stage1[r-1]
                thinking1, answer1 = await agent(input_infos_1, debate_instr_stage1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking1.content}; answer: {answer1.content}")
            all_thinking_stage1[r].append(thinking1)
            all_answer_stage1[r].append(answer1)
    final_decision_agent_stage1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_stage1([taskInfo] + all_thinking_stage1[-1] + all_answer_stage1[-1], "Sub-task 1: Synthesize and finalize extraction of parameters and problem setup." + " Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Stage 1 Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_stage2 = "Sub-task 2: Based on the output from Stage 1 Sub-task 1, formulate the electrostatic boundary conditions imposed by the uncharged spherical conductor with an off-center cavity containing charge +q. Emphasize that the conductor's shielding effect and induced surface charges must be accounted for, and that the external field cannot be treated as simply due to the charge at the cavity center. Identify the necessity of using the method of images or equivalent potential theory to solve for the induced charges and effective external field source."
    N_sc = self.max_sc
    cot_agents_stage2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_stage2 = []
    possible_thinkings_stage2 = []
    subtask_desc2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": cot_sc_instruction_stage2,
        "context": ["user query", thinking1, answer1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_stage2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_stage2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_stage2[i].id}, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_stage2.append(answer2)
        possible_thinkings_stage2.append(thinking2)
    final_decision_agent_stage2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_stage2([taskInfo, thinking1, answer1] + possible_thinkings_stage2 + possible_answers_stage2, "Sub-task 2: Synthesize and choose the most consistent and correct formulation of boundary conditions." , is_sub_task=True)
    sub_tasks.append(f"Stage 1 Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    reflect_inst_stage3 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_stage3 = "Sub-task 3: Apply the method of images or an equivalent electrostatic technique to explicitly derive the induced charge distribution on the conductor's inner and outer surfaces due to the off-center cavity charge +q. Derive the positions and magnitudes of image charges required to satisfy the boundary conditions, ensuring the displacement s and angle theta are incorporated. Demonstrate how the conductor's shielding relocates the effective external source to the conductor's center, not the cavity center, and derive the resulting expression for the electric field magnitude at point P outside the conductor." + reflect_inst_stage3
    cot_agent_stage3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_stage3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_stage3 = self.max_round
    cot_inputs_stage3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": cot_reflect_instruction_stage3,
        "context": ["user query", thinking1, answer1, thinking2, answer2],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_stage3(cot_inputs_stage3, cot_reflect_instruction_stage3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_stage3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_stage3):
        feedback, correct = await critic_agent_stage3([taskInfo, thinking3, answer3], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_stage3.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_stage3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_stage3(cot_inputs_stage3, cot_reflect_instruction_stage3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_stage3.id}, refining thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Stage 2 Sub-task 1 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    debate_instr_stage4 = "Sub-task 4: Critically evaluate the derived electric field expression against the provided multiple-choice options. Verify that the expression correctly depends on the distance L from the conductor center to point P, reflecting the conductor's shielding and induced charges, rather than on l or the cavity center. Select the correct formula that matches the physical and mathematical analysis, explicitly justifying why other options fail due to ignoring induced charges or incorrect distance dependence. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_stage4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_stage4 = self.max_round
    all_thinking_stage4 = [[] for _ in range(N_max_stage4)]
    all_answer_stage4 = [[] for _ in range(N_max_stage4)]
    subtask_desc4 = {
        "subtask_id": "stage_3.subtask_1",
        "instruction": debate_instr_stage4,
        "context": ["user query", thinking3, answer3],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_stage4):
        for i, agent in enumerate(debate_agents_stage4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], debate_instr_stage4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3, answer3] + all_thinking_stage4[r-1] + all_answer_stage4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instr_stage4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking_stage4[r].append(thinking4)
            all_answer_stage4[r].append(answer4)
    final_decision_agent_stage4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_stage4([taskInfo, thinking3, answer3] + all_thinking_stage4[-1] + all_answer_stage4[-1], "Sub-task 4: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Stage 3 Sub-task 1 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs
