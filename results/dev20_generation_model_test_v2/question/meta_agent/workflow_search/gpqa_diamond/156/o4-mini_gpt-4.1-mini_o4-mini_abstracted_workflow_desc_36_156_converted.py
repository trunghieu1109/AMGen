async def forward_156(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    N_sc = self.max_sc
    cot_sc_instruction = (
        "Sub-task 0_1: Extract and summarize the virological and clinical context: "
        "characterize the pathogen as a retrovirus, define diagnostic speed and accuracy requirements, "
        "and note sample types."
    )
    cot_sc_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent",
                                  model=self.node_model, temperature=0.5)
                    for _ in range(N_sc)]
    possible_thinkings0 = []
    possible_answers0 = []
    subtask_desc0 = {
        "subtask_id": "subtask_0_1",
        "instruction": cot_sc_instruction,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for agent in cot_sc_agents:
        thinking_i, answer_i = await agent([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_thinkings0.append(thinking_i)
        possible_answers0.append(answer_i)
    final_decision0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent",
                                   model=self.node_model, temperature=0.0)
    synth_instr0 = (
        "Sub-task 0_1: Synthesize and choose the most consistent summary for context extraction."
    )
    thinking0_1, answer0_1 = await final_decision0(
        [taskInfo] + possible_thinkings0 + possible_answers0,
        synth_instr0,
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 0_1 output: thinking - {thinking0_1.content}; answer - {answer0_1.content}")
    subtask_desc0['response'] = {"thinking": thinking0_1, "answer": answer0_1}
    logs.append(subtask_desc0)
    print("Step 0_1: ", sub_tasks[-1])

    debate_instr = (
        "Given solutions to the problem from other agents, consider their opinions as additional advice. "
        "Please think carefully and provide an updated answer."
    )
    debate_instruction_1 = (
        "Sub-task 1_1: Evaluate nucleic acid–based approaches: compare feasibility and specificity of direct sequencing vs cDNA sequencing, "
        "and assess conventional, nested, and real-time PCR for speed, sensitivity, and contamination risk."
        + debate_instr
    )
    debate_agents1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent",
                                    model=self.node_model, role=role, temperature=0.5)
                     for role in self.debate_role]
    rounds1 = self.max_round
    all_thinking1 = [[] for _ in range(rounds1)]
    all_answer1 = [[] for _ in range(rounds1)]
    subtask_desc1 = {
        "subtask_id": "subtask_1_1",
        "instruction": debate_instruction_1,
        "context": ["user query", thinking0_1, answer0_1],
        "agent_collaboration": "Debate"
    }
    for r in range(rounds1):
        for agent in debate_agents1:
            if r == 0:
                thinking1, answer1 = await agent([taskInfo, thinking0_1, answer0_1], debate_instruction_1, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking0_1, answer0_1] + all_thinking1[r-1] + all_answer1[r-1]
                thinking1, answer1 = await agent(inputs, debate_instruction_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking1.content}; answer: {answer1.content}")
            all_thinking1[r].append(thinking1)
            all_answer1[r].append(answer1)
    final_decision1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent",
                                   model=self.node_model, temperature=0.0)
    synth_instr1 = (
        "Sub-task 1_1: Given all the above thinking and answers, reason over them carefully and provide a final evaluation of nucleic acid methods."
    )
    thinking1_1, answer1_1 = await final_decision1(
        [taskInfo, thinking0_1, answer0_1] + all_thinking1[-1] + all_answer1[-1],
        synth_instr1,
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1_1 output: thinking - {thinking1_1.content}; answer - {answer1_1.content}")
    subtask_desc1['response'] = {"thinking": thinking1_1, "answer": answer1_1}
    logs.append(subtask_desc1)
    print("Step 1_1: ", sub_tasks[-1])

    reflect_inst = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot_reflect_instruction = (
        "Sub-task 2_1: Integrate the assessments to recommend the optimal molecular diagnostic protocol (target, sequencing, amplification method), "
        "justify the choice, and outline key design steps for the kit."
        + reflect_inst
    )
    cot_reflect_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent",
                                      model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent",
                                model=self.node_model, temperature=0.0)
    cot_inputs = [taskInfo, thinking0_1, answer0_1, thinking1_1, answer1_1]
    subtask_desc2 = {
        "subtask_id": "subtask_2_1",
        "instruction": cot_reflect_instruction,
        "context": [taskInfo, thinking0_1, answer0_1, thinking1_1, answer1_1],
        "agent_collaboration": "Reflexion"
    }
    thinking2_1, answer2_1 = await cot_reflect_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_reflect_agent.id}, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent([taskInfo, thinking2_1, answer2_1],
                                              "Please review the answer above and criticize on where might be wrong. "
                                              "If you are absolutely sure it is correct, output exactly 'True' in 'correct'",
                                              i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs.extend([thinking2_1, answer2_1, feedback])
        thinking2_1, answer2_1 = await cot_reflect_agent(cot_inputs, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_reflect_agent.id}, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
    sub_tasks.append(f"Sub-task 2_1 output: thinking - {thinking2_1.content}; answer - {answer2_1.content}")
    subtask_desc2['response'] = {"thinking": thinking2_1, "answer": answer2_1}
    logs.append(subtask_desc2)
    print("Step 2_1: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking2_1, answer2_1, sub_tasks, agents)
    return final_answer, logs