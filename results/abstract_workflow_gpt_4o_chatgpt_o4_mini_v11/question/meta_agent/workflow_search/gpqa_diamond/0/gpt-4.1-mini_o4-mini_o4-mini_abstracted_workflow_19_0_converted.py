async def forward_0(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    """
    [Stage 1: Collect and Verify Molecular Structures and Symmetry Data]
    [Objective]
    - Collect detailed 3D molecular structures and known symmetry information for each molecule: triisopropyl borate, quinuclidine, benzo[1,2-c:3,4-c:5,6-c]trifuran-1,3,4,6,7,9-hexaone, and triphenyleno[1,2-c:5,6-c:9,10-c]trifuran-1,3,6,8,11,13-hexaone.
    - Retrieve experimentally or computationally determined geometries and any reported point group symmetries from authoritative chemical databases or literature, ensuring data accuracy and traceability.
    [Agent Collaborations]
    - Use Chain-of-Thought (CoT) agent to carefully gather and verify molecular data for each molecule.
    """

    cot_instruction_1 = (
        "Sub-task 1: Collect and verify detailed 3D molecular structures and known symmetry information "
        "for triisopropyl borate, quinuclidine, benzo[1,2-c:3,4-c:5,6-c]trifuran-1,3,4,6,7,9-hexaone, and "
        "triphenyleno[1,2-c:5,6-c:9,10-c]trifuran-1,3,6,8,11,13-hexaone, including geometries and reported point group symmetries "
        "from authoritative sources, ensuring data accuracy and traceability."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, collecting and verifying molecular data, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Subtask 1 answer: ", sub_tasks[-1])

    """
    [Stage 2: Define and Clarify C3h Symmetry Criteria]
    [Objective]
    - Define and clarify the criteria for a molecule to have C3h symmetry by explicitly describing the symmetry elements involved (a C3 principal rotation axis and a horizontal mirror plane σh).
    - Emphasize that the presence of additional symmetry elements does not exclude a molecule from possessing C3h symmetry.
    - Establish clear, inclusive criteria for identifying C3h symmetry elements in molecular structures to guide subsequent analyses.
    [Agent Collaborations]
    - Use Self-Consistency Chain-of-Thought (SC-CoT) to generate and verify inclusive criteria for C3h symmetry.
    """

    cot_sc_instruction_2 = (
        "Sub-task 2: Define and clarify the criteria for a molecule to have C3h symmetry, explicitly describing the symmetry elements involved, "
        "including a C3 principal rotation axis and a horizontal mirror plane σh, and emphasizing that additional symmetry elements do not exclude C3h symmetry."
    )
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, defining C3h symmetry criteria, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    most_common_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[most_common_answer_2]
    answer2 = answermapping_2[most_common_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {
        "thinking": thinking2,
        "answer": answer2
    }
    logs.append(subtask_desc2)
    print("Subtask 2 answer: ", sub_tasks[-1])

    """
    [Stage 3: Analyze Each Molecule for C3h Symmetry]
    [Objective]
    - Using the verified molecular structure and symmetry data from Subtask 1 and the inclusive C3h criteria from Subtask 2, analyze each molecule (triisopropyl borate, quinuclidine, benzo[1,2-c:3,4-c:5,6-c]trifuran-1,3,4,6,7,9-hexaone, triphenyleno[1,2-c:5,6-c:9,10-c]trifuran-1,3,6,8,11,13-hexaone) to determine whether it possesses the C3 principal axis and horizontal mirror plane σh.
    - Explicitly cite the structural features and symmetry elements identified.
    [Agent Collaborations]
    - Use Reflexion pattern to analyze each molecule in turn, refining the analysis based on feedback.
    """

    molecules = [
        "triisopropyl borate",
        "quinuclidine",
        "benzo[1,2-c:3,4-c:5,6-c]trifuran-1,3,4,6,7,9-hexaone",
        "triphenyleno[1,2-c:5,6-c:9,10-c]trifuran-1,3,6,8,11,13-hexaone"
    ]

    analysis_results = {}

    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round

    for idx, molecule in enumerate(molecules, start=3):
        cot_reflect_instruction = (
            f"Sub-task {idx}: Analyze {molecule} to determine if it possesses the C3 principal axis and horizontal mirror plane σh, "
            "explicitly citing structural features and symmetry elements identified, based on verified molecular data and C3h criteria."
        )
        cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
        thinking, answer = await cot_agent_3(cot_inputs + [molecule], cot_reflect_instruction, 0, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, initial analysis of {molecule}, thinking: {thinking.content}; answer: {answer.content}")

        for i in range(N_max_3):
            feedback, correct = await critic_agent_3([taskInfo, thinking, answer, molecule],
                                                   f"Please review the analysis of {molecule} for C3h symmetry and provide limitations.",
                                                   i, is_sub_task=True)
            agents.append(f"Critic agent {critic_agent_3.id}, feedback on {molecule}, thinking: {feedback.content}; answer: {correct.content}")
            if correct.content == "True":
                break
            cot_inputs.extend([thinking, answer, feedback])
            thinking, answer = await cot_agent_3(cot_inputs + [molecule], cot_reflect_instruction, i + 1, is_sub_task=True)
            agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refined analysis of {molecule}, thinking: {thinking.content}; answer: {answer.content}")

        analysis_results[molecule] = answer.content
        sub_tasks.append(f"Sub-task {idx} output: thinking - {thinking.content}; answer - {answer.content}")
        logs.append({
            "subtask_id": f"subtask_{idx}",
            "instruction": cot_reflect_instruction,
            "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
            "agent_collaboration": "Reflexion",
            "response": {
                "thinking": thinking,
                "answer": answer
            }
        })
        print(f"Subtask {idx} answer: ", sub_tasks[-1])

    """
    [Stage 4: Synthesize and Determine Final C3h Symmetry Result]
    [Objective]
    - Synthesize and compare the symmetry analyses from Stage 3 for all molecules.
    - Explicitly reference identified symmetry elements and any additional features.
    - Determine which molecule(s) possess C3h symmetry according to the inclusive criteria.
    - Provide a concise, unambiguous final answer by selecting the correct multiple-choice letter (A, B, C, or D) corresponding to the molecule(s) with C3h symmetry.
    [Agent Collaborations]
    - Use Debate pattern among agents representing each molecule's analysis to reach consensus on final answer.
    """

    debate_instruction_4 = (
        "Sub-task 7: Synthesize and compare the symmetry analyses of all four molecules, "
        "explicitly referencing identified symmetry elements and additional features, to determine which molecule(s) possess C3h symmetry according to the inclusive criteria. "
        "Provide a concise, unambiguous final answer as a single letter (A, B, C, or D) corresponding to the molecule(s) with C3h symmetry."
    )

    debate_roles = [
        "Agent A (triisopropyl borate)",
        "Agent B (quinuclidine)",
        "Agent C (benzo[1,2-c:3,4-c:5,6-c]trifuran-1,3,4,6,7,9-hexaone)",
        "Agent D (triphenyleno[1,2-c:5,6-c:9,10-c]trifuran-1,3,6,8,11,13-hexaone)"
    ]

    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    N_max_4 = self.max_round
    all_thinking_4 = [[] for _ in range(N_max_4)]
    all_answer_4 = [[] for _ in range(N_max_4)]

    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                input_infos_4 = [taskInfo] + [analysis_results[mol] for mol in molecules]
                thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo] + [analysis_results[mol] for mol in molecules] + all_thinking_4[r-1] + all_answer_4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating final C3h symmetry decision, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking_4[r].append(thinking4)
            all_answer_4[r].append(answer4)

    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_final, answer_final = await final_decision_agent_4([taskInfo] + all_thinking_4[-1] + all_answer_4[-1],
                                                              "Sub-task 7: Make final decision on which molecule(s) have C3h symmetry, outputting a single letter (A, B, C, or D).",
                                                              is_sub_task=True)
    agents.append(f"Final Decision agent, determining final C3h symmetry answer, thinking: {thinking_final.content}; answer: {answer_final.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking_final.content}; answer - {answer_final.content}")
    subtask_desc_final = {
        "subtask_id": "subtask_7",
        "instruction": debate_instruction_4,
        "context": ["user query"] + [analysis_results[mol] for mol in molecules],
        "agent_collaboration": "Debate",
        "response": {
            "thinking": thinking_final,
            "answer": answer_final
        }
    }
    logs.append(subtask_desc_final)
    print("Subtask 7 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_final, answer_final, sub_tasks, agents)
    return final_answer, logs
