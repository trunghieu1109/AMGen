async def forward_8(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Identify and gather structural and symmetry-related information for each molecule
    cot_instruction_1 = (
        "Sub-task 1: Identify and gather structural and symmetry-related information for each given molecule: "
        "triisopropyl borate, quinuclidine, benzo[1,2-c:3,4-c:5,6-c]trifuran-1,3,4,6,7,9-hexaone, and "
        "triphenyleno[1,2-c:5,6-c:9,10-c]trifuran-1,3,6,8,11,13-hexaone. Include molecular geometry, point group symmetry, and known symmetry elements."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, gathering structural and symmetry info, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Stage 1 subtasks 2 to 5: Analyze each molecule for C3h symmetry using Self-Consistency Chain-of-Thought
    molecules = [
        "triisopropyl borate",
        "quinuclidine",
        "benzo[1,2-c:3,4-c:5,6-c]trifuran-1,3,4,6,7,9-hexaone",
        "triphenyleno[1,2-c:5,6-c:9,10-c]trifuran-1,3,6,8,11,13-hexaone"
    ]

    sc_cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]

    analysis_results = {molecule: [] for molecule in molecules}

    for idx, molecule in enumerate(molecules):
        sc_instruction = (
            f"Sub-task {idx+2}: Analyze the molecular geometry and symmetry elements of {molecule} "
            "based on the information gathered in Sub-task 1 to determine if it exhibits C3h symmetry. "
            "Consider all possible structural interpretations and symmetry elements."
        )
        possible_answers = []
        thinkingmapping = {}
        answermapping = {}
        for i in range(self.max_sc):
            thinking_i, answer_i = await sc_cot_agents[i]([taskInfo, thinking1, answer1], sc_instruction, is_sub_task=True)
            agents.append(f"CoT-SC agent {sc_cot_agents[i].id}, analyzing {molecule} for C3h symmetry, thinking: {thinking_i.content}; answer: {answer_i.content}")
            possible_answers.append(answer_i.content)
            thinkingmapping[answer_i.content] = thinking_i
            answermapping[answer_i.content] = answer_i
        # Majority vote on answers
        answer_counts = Counter(possible_answers)
        most_common_answer, _ = answer_counts.most_common(1)[0]
        analysis_results[molecule] = (thinkingmapping[most_common_answer], answermapping[most_common_answer])
        sub_tasks.append(f"Sub-task {idx+2} output: thinking - {thinkingmapping[most_common_answer].content}; answer - {most_common_answer}")
        print(f"Step {idx+2}: ", sub_tasks[-1])

    # Stage 2: Compare the symmetry analyses from subtasks 2 to 5 to identify which molecule(s) exhibit C3h symmetry
    compare_instruction = (
        "Sub-task 6: Compare the symmetry analyses from Sub-tasks 2 to 5 to identify which molecule(s) exhibit C3h symmetry. "
        "Summarize the findings and highlight molecules with confirmed C3h symmetry."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    inputs_6 = [taskInfo]
    for molecule in molecules:
        thinking_mol, answer_mol = analysis_results[molecule]
        inputs_6.extend([thinking_mol, answer_mol])
    thinking6, answer6 = await cot_agent_6(inputs_6, compare_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, comparing symmetry analyses, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    # Stage 2: Select and confirm the molecule with C3h symmetry based on comparison
    confirm_instruction = (
        "Sub-task 7: Select and confirm the molecule from the given choices that has C3h symmetry based on the comparison and analysis results from Sub-task 6. "
        "Provide a clear final answer naming the molecule(s) with C3h symmetry."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6, answer6], confirm_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, confirming molecule with C3h symmetry, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer
